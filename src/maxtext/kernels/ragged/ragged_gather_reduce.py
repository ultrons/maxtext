# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ragged gather reduce kernel implementation from tpu-inference."""
# Source from experimental/users/kyuyeunk/vllm/kernels/sparse_core/ragged_gather_reduce.py

import functools
import math
import jax
from jax._src import core as jax_core
from jax._src import tree_util as _tree_util
from jax._src.pallas import core as pl_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
from packaging.version import Version


# JAX <= 0.10.0 used `out_shape`/`scratch_shapes` kwargs for `pl.kernel`; later
# versions renamed them to `out_type`/`scratch_types`.
if Version(jax.__version__) <= Version("0.10.0"):
  _OUT_KW = "out_shape"
  _SCRATCH_KW = "scratch_shapes"
  _COMPILER_PARAMS = {
      "use_tc_tiling_on_sc": True,
      "disable_bounds_checks": True,
  }
else:
  _OUT_KW = "out_type"
  _SCRATCH_KW = "scratch_types"
  _COMPILER_PARAMS = {
      "use_tc_tiling_on_sc": True,
      "disable_bounds_checks": True,
      "needs_layout_passes": False,
  }


# ceil up to the nearest multiple of b.
def _align_to(a, b):
  return ((a + b - 1) // b) * b


def get_cost_estimate(
    padded_input_size: int,
    aligned_hidden_size: int,
    reduce_group_size: int,
    input_dtype_bytes: int,
    bytes_accessed_override: int = -1,
    flops_override: int = -1,
) -> pl.CostEstimate:
  """Returns a cost estimate for the ragged gather-reduce kernel.

  The kernel gathers rows, multiplies each by a scalar weight, and reduces
  (sums) every ``reduce_group_size`` rows into one output row.

  Args:
    padded_input_size: Total number of source rows (after padding).
    aligned_hidden_size: Number of columns (after alignment).
    reduce_group_size: Number of source rows reduced into each output row.
    input_dtype_bytes: Size of one input element in bytes.
    bytes_accessed_override: If > 0, use this value as bytes_accessed instead
      of auto-computing.  -1 (default) means auto-compute.
    flops_override: If > 0, use this value as the flop count instead of
      auto-computing.  -1 (default) means auto-compute.

  Returns:
    A ``pl.CostEstimate`` suitable for XLA scheduling.
  """
  # Flops:
  #   - one multiply per element for weighting: padded_input_size * aligned_hidden_size
  #   - one add per element for reduction:       padded_input_size * aligned_hidden_size
  if flops_override > 0:
    flops = flops_override
  else:
    flops = 2 * padded_input_size * aligned_hidden_size

  if bytes_accessed_override > 0:
    bytes_accessed = bytes_accessed_override
  else:
    # Bytes accessed:
    #   read  – input rows + src_indices (int32) + dst_indices (int32) + topk_weights (f32)
    #   write – output rows (float32)
    bytes_in = padded_input_size * aligned_hidden_size * input_dtype_bytes  # input rows
    bytes_in += padded_input_size * 4  # src_indices (int32)
    bytes_in += padded_input_size * 4  # dst_indices (int32)
    bytes_in += padded_input_size * 4  # topk_weights (float32)
    output_rows = padded_input_size // reduce_group_size
    bytes_out = output_rows * aligned_hidden_size * 4  # output rows (float32)
    bytes_accessed = bytes_in + bytes_out

  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=bytes_accessed,
      transcendentals=0,
  )


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Fallback to JAX implementation."""
  out = x[indices] * topk_weights[:, None].astype(jnp.float32)
  out = jnp.where(valid_rows_mask[:, None], out, 0)
  out = out.reshape(-1, reduce_group_size, out.shape[-1])
  out = jnp.sum(out, axis=1).astype(x.dtype)
  return out


def main_kernel(
    # Inputs.
    num_rows_per_row_partition_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    num_rows_per_row_partition_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    prev_iter_last_row_vmem_ref: jax.Ref,
    src_indices_vmem_ref: jax.Ref,
    dst_indices_vmem_ref: jax.Ref,
    topk_weights_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
):
  """Main Pallas kernel for ragged gather and reduction on SparseCore."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_simd_lanes = sc_info.num_lanes
  num_lanes = tpu_info.num_lanes
  col_size = out_vmem_ref.shape[-1]
  num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))

  recv_sem = sem_ref.at[0]
  send_sem = sem_ref.at[1]

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_cores,),
      core_axis_name=(core_axis_name, subcore_axis_name),
      dimension_semantics=(pltpu.PARALLEL,),
  )
  def inner_kernel():
    core_id = pl.program_id(0)
    # Partition stride over the SLOT arrays (src/dst indices, weights) must be
    # derived from the indices operand itself, NOT from the input buffer: the
    # chunked combine (decouple_combine_rs_chunks) passes the FULL expert-sorted
    # buffer with a per-chunk SLICE of the indices, so in_hbm_ref.shape[0] can be
    # N x larger than the slot arrays. Deriving the stride from in_hbm_ref made
    # every row partition p >= 1 read src/dst/weights OUT OF BOUNDS (silently,
    # with disable_bounds_checks=True) and scatter garbage -- for EVERY n_chunks
    # > 1. For the un-chunked call the two are equal, so this is a no-op there.
    row_partition_size = src_indices_hbm_ref.shape[0] // num_row_partitions
    row_partition_id = core_id // num_column_partitions
    col_partition_id = core_id % num_column_partitions

    # Read total number of valid source rows for the current row partition.
    dma = pltpu.make_async_copy(
        num_rows_per_row_partition_ref.at[pl.ds(0, num_simd_lanes)],
        num_rows_per_row_partition_vmem_ref,
        recv_sem,
    )
    dma.start()
    dma.wait()
    num_rows_current_row_partition = jnp.array(0, jnp.int32)
    num_rows_per_row_partition = num_rows_per_row_partition_vmem_ref[...]
    for i in range(num_row_partitions):
      num_rows_current_row_partition = jnp.where(
          row_partition_id == i,
          num_rows_per_row_partition[i],
          num_rows_current_row_partition,
      )

    row_tile_size = num_simd_lanes
    num_row_tiles = pl.cdiv(num_rows_current_row_partition, row_tile_size)
    row_start = row_partition_id * row_partition_size
    col_start = col_partition_id * col_size

    @pl.loop(0, num_row_tiles)
    def row_loop(row_block_id):
      row_tile_start = row_start + row_block_id * num_simd_lanes
      # The destination row from the last source row in the previous row tile,
      # retrieve it before DMA the new data into `dst_indices_vmem_ref`.
      prev_dst_row_hbm = jnp.where(row_block_id == 0, -1, dst_indices_vmem_ref[...][num_simd_lanes - 1])
      dma_list = []
      dma_list.append(
          pltpu.make_async_copy(
              src_indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              src_indices_vmem_ref,
              recv_sem,
          )
      )
      dma_list.append(
          pltpu.make_async_copy(
              dst_indices_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              dst_indices_vmem_ref,
              recv_sem,
          )
      )
      dma_list.append(
          pltpu.make_async_copy(
              topk_weights_hbm_ref.at[pl.ds(row_tile_start, num_simd_lanes)],
              topk_weights_vmem_ref,
              recv_sem,
          )
      )
      jax.tree.map(lambda x: x.start(), dma_list)
      jax.tree.map(lambda x: x.wait(), dma_list)

      # HBM to VMEM transfer.
      src_indices = src_indices_vmem_ref[...]
      dst_indices = dst_indices_vmem_ref[...]
      topk_weights = topk_weights_vmem_ref[...]

      in_dtype = in_hbm_ref.dtype
      input_dtype_bits = jax.dtypes.itemsize_bits(in_dtype)
      input_packing = 32 // input_dtype_bits

      in_32b_hbm_ref = in_hbm_ref.bitcast(jnp.uint32)
      out_32b_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

      for col_vmem_start in range(0, col_size, num_lanes):
        col_hbm_start = col_start + col_vmem_start
        for row_vmem in range(num_simd_lanes):
          row_hbm = src_indices[row_vmem] // input_packing
          # Since we have changed layout from (8, 128) to (1, 128), continuous
          # memory address does not yield desired values anymore. Therefore,
          # we break up a dmas into multiple num_lanes sized requests.
          pltpu.make_async_copy(
              in_32b_hbm_ref.at[row_hbm, pl.ds(col_hbm_start, num_lanes)],
              out_vmem_ref.at[row_vmem, pl.ds(col_vmem_start, num_lanes)],
              recv_sem,
          ).start()

      # VMEM to HBM transfer.
      # Use dynamic loop to minimize register spills.
      @pl.loop(0, col_size, step=num_lanes, init_carry=(prev_dst_row_hbm,))
      @jax.named_scope("dma_write_loop")
      def dma_write_loop(col_vmem_start, carry):
        col_hbm_start = col_start + col_vmem_start

        for _ in range(num_simd_lanes):
          pltpu.make_async_copy(
              in_32b_hbm_ref.at[0, :num_lanes],
              out_vmem_ref.at[0, :num_lanes],
              recv_sem,
          ).wait()

        for col_compute_offset in range(0, num_lanes, num_simd_lanes):
          col_slice = pl.ds(col_vmem_start + col_compute_offset, num_simd_lanes)

          previous_accumulated_data = None
          for row_src in range(num_simd_lanes):
            row_src_pack = src_indices[row_src] % input_packing

            # Load data from vmem.
            data = out_vmem_ref[row_src, col_slice]

            # Extract data and cast to float32.
            if in_dtype == jnp.bfloat16:
              data = jnp.bitwise_left_shift(data, jnp.where(row_src_pack == 0, 16, 0))
              # Mask out the lower 16 bits. -65536 is 0xffff0000
              data = jnp.bitwise_and(data, -65536)
              data = jax.lax.bitcast_convert_type(data, jnp.float32)
            elif in_dtype == jnp.float32:
              data = jax.lax.bitcast_convert_type(data, jnp.float32)
            else:
              raise ValueError(
                  f"Dtype {in_dtype} is not yet supported for ragged data extraction. Supported dtypes: bfloat16, float32."
              )
            # Accumulate at float32 precision
            data = data * topk_weights[row_src]

            dst_row_hbm = dst_indices[row_src]
            if row_src == 0:
              # carry[0] is the last dst_row_hbm from the previous row_tile.
              prev_row_hbm = carry[0]
              previous_accumulated_data = jax.lax.bitcast_convert_type(
                  prev_iter_last_row_vmem_ref[0, col_slice], jnp.float32
              )
            else:
              prev_row_hbm = dst_indices[row_src - 1]
              assert previous_accumulated_data is not None

            # We guarantee source rows that contribute to the same destination
            # row are adjacent to each other in the order of being processed.
            # If the current src row contributes to the same destination row as
            # the previous src row, we accumulate the data with the previously
            # accumulated data.
            accumulated_data = jnp.where(
                dst_row_hbm == prev_row_hbm,
                previous_accumulated_data + data,
                data,
            )
            previous_accumulated_data = accumulated_data
            data_to_write = jax.lax.bitcast_convert_type(accumulated_data, jnp.uint32)
            out_vmem_ref[row_src, col_slice] = data_to_write

            # We write the last row (within a row tile)'s accumulated data to
            # the prev_iter_last_row_vmem_ref. If the first src row in the next
            # row_tile contributes to the same destination row as the last src
            # row in the current row_tile, the latest accumulated data in
            # prev_iter_last_row_vmem_ref will get used.
            if row_src == num_simd_lanes - 1:
              prev_iter_last_row_vmem_ref[0, col_slice] = data_to_write

        # Start dma write.
        # When there are multiple sources rows in the current row_tile that
        # contribute to the same destination row, the accumulated data is
        # stored in the last row's idx in `out_vmem_ref`.
        # Logically, we could skip all the source rows that are not the last
        # for each destination row, but we want to avoid using `pl.when` for
        # efficiency. We just repeat the write of latest accumulated data
        # multiple times.
        # `src_row_idx_in_vmem` tracks the right idx in vmem for each hbm write.
        src_row_idx_in_vmem = []
        row_valid_vec = []
        for row_vmem_idx in reversed(range(num_simd_lanes)):
          row_valid = row_block_id * num_simd_lanes + row_vmem_idx < num_rows_current_row_partition
          row_valid_vec.append(row_valid)
          if row_vmem_idx == num_simd_lanes - 1:
            src_row_idx_in_vmem.append(row_vmem_idx)
          else:
            next_row_valid = row_valid_vec[-2]
            src_row_idx_in_vmem.append(
                jnp.where(
                    jnp.logical_and(
                        next_row_valid,
                        (dst_indices[row_vmem_idx] == dst_indices[row_vmem_idx + 1]),
                    ),
                    src_row_idx_in_vmem[-1],
                    row_vmem_idx,
                )
            )
        src_row_idx_in_vmem.reverse()
        row_valid_vec.reverse()

        # There must be at least one valid row to write in the current row_tile.
        # When num valid writes is not a multiple of row_tile_size, we repeat
        # the last valid write to avoid valid data in hbm being overwritten
        # (and to avoid using `pl.when`).
        last_valid_src_row_vmem = -1
        last_valid_dst_row_hbm = -1
        for i, (src_row_idx_in_vmem, row_valid) in enumerate(zip(src_row_idx_in_vmem, row_valid_vec, strict=True)):
          src_row_vmem = jnp.where(row_valid, src_row_idx_in_vmem, last_valid_src_row_vmem)
          dst_row_hbm = jnp.where(row_valid, dst_indices[i], last_valid_dst_row_hbm)
          pltpu.make_async_copy(
              out_vmem_ref.at[src_row_vmem, pl.ds(col_vmem_start, num_lanes)],
              out_32b_hbm_ref.at[dst_row_hbm, pl.ds(col_hbm_start, num_lanes)],
              send_sem,
          ).start()
          last_valid_src_row_vmem = src_row_vmem
          last_valid_dst_row_hbm = dst_row_hbm

        return carry

      # Wait for dma write to finish.
      for _ in range(0, col_size, num_lanes):
        for _ in range(num_simd_lanes):
          pltpu.make_async_copy(
              out_vmem_ref.at[0, :num_lanes],
              out_32b_hbm_ref.at[0, :num_lanes],
              send_sem,
          ).wait()

  inner_kernel()


def _preprocess(
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    num_row_partitions: int,
    num_simd_lanes: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Preprocesses indices for ragged gather reduce."""
  assert indices.ndim == 1, "Ragged scatter only supports 1d indices."

  row_partition_size = indices.shape[0] // num_row_partitions
  valid_rows_mask = valid_rows_mask.reshape(num_row_partitions, -1)

  # Move all the valid source rows to the beginning of each row partition.
  sorted_by_validity = jnp.argsort(~valid_rows_mask, descending=False, stable=True, axis=-1)
  sorted_by_validity += (
      jnp.broadcast_to(
          jnp.arange(num_row_partitions)[:, None],
          (num_row_partitions, row_partition_size),
      )
      * row_partition_size
  )
  sorted_by_validity = sorted_by_validity.reshape(-1)

  src_indices = indices[sorted_by_validity]
  # `reduce_group_size` source rows are mapped (and reduced) to the same output
  # row.
  dst_indices = sorted_by_validity // reduce_group_size
  topk_weights = topk_weights[sorted_by_validity]
  topk_weights = topk_weights.astype(jnp.float32)

  num_src_rows_per_row_partition = jnp.sum(valid_rows_mask, axis=-1)
  assert num_row_partitions <= num_simd_lanes
  num_src_rows_per_row_partition = jnp.pad(
      num_src_rows_per_row_partition.astype(jnp.int32),
      (0, num_simd_lanes - num_row_partitions),
  )
  # If there is no valid source row in a reduce group, we set the mask to
  # False, so that the output for that group is set to zero.
  mask = jnp.any(valid_rows_mask.reshape(-1, reduce_group_size), axis=-1)

  return (
      src_indices,
      dst_indices,
      topk_weights,
      num_src_rows_per_row_partition,
      mask,
  )


@functools.partial(
    jax.jit, static_argnames=("reduce_group_size", "enforce_fallback", "flops_override", "bytes_accessed_override")
)
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    enforce_fallback: bool = False,
    flops_override: int = -1,
    bytes_accessed_override: int = -1,
) -> jax.Array:
  """Gathers `x` according to `indices`, applies weights and masks, and reduces.

  This function performs a gathered lookup from `x` using `indices`, scales the
  obtained rows by `topk_weights`, masks out any rows where `valid_rows_mask` is
  False, and then groups every `reduce_group_size` rows together and reduces
  them via summation.

  The typical use case of this kernel is unpermute + local-reduction in the
  MOE after GMM. Compared to maxtext.src.maxtext.kernels.gather_reduce_sc,
  this kernel provides better performance if large sparsity exists in
  `valid_rows_mask`. For example, expert_parallelism =8, 16 etc.

  Args:
    x: A 2D JAX array of input features with shape `(input_size, hidden_size)`.
    indices: A 1D JAX array of indices to gather with shape `(input_size,)`.
    topk_weights: A 1D JAX array of weights to scale the gathered rows with
      shape `(input_size,)`.
    valid_rows_mask: A 1D boolean JAX array indicating which gathered rows are
      valid, with shape `(input_size,)`.
    reduce_group_size: An integer representing the number of consecutive rows to
      reduce (sum) together.
    enforce_fallback: Static bool flag. When ``True``, unconditionally use the
      JAX reference implementation instead of the SparseCore kernel.
      When ``False`` (default), use the SparseCore kernel and raise any error.

  Returns:
    A 2D JAX array of reduced data with shape
    `(input_size // reduce_group_size, hidden_size)`.
  """

  assert x.ndim == 2, "ragged_gather_reduce only supports 2d inputs."
  assert indices.ndim == 1, "ragged_gather_reduce only supports 1d indices."
  assert topk_weights.ndim == 1, "ragged_gather_reduce only supports 1d topk_weights."
  assert valid_rows_mask.ndim == 1, "ragged_gather_reduce only supports 1d valid_rows_mask."

  if enforce_fallback:
    # Fallback is enforced. Use JAX reference. (Checked BEFORE get_tpu_info(),
    # which raises on non-TPU backends -- keeps the pure-JAX path CPU-testable.)
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    # Sparse core is not available. Use JAX reference.
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  # Heuristic threshold on whether to fallback for small inputs.
  dtype = x.dtype
  dtype_bytes = jax.dtypes.itemsize_bits(dtype) // 8

  hidden_size = x.shape[-1]
  input_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores

  # This kernel partitions the output's columns into `num_column_partitions`
  # and partition the output's rows into `num_row_partitions` and run each
  # {row_partition} x {column_partition} combination on a separate SC subcore
  # for parallelism. With such work partitioning, we guarantee that there
  # won't be write collision (from different subcores) to any output row X
  # column.
  #
  # Each column partition should be multiple of 128 (number of lanes) due to
  # DMA requirements. Unless requiring padding on the column dimension, larger
  # column partitions (thus smaller row partitions given fixed num_cores) is
  # more preferable because large row partition may lead to imbalanced load
  # (valid_rows_mask may have more rows in some partitions than others).
  # Most LLM's hidden size is multiple of 1024, `num_column_partitions=8`
  # should work well in practice without requiring padding on the column size.
  num_column_partitions = 8
  assert num_cores % num_column_partitions == 0
  num_rows_partitions = num_cores // num_column_partitions

  aligned_hidden_size = _align_to(hidden_size, 128 * num_column_partitions)
  col_size = aligned_hidden_size // num_column_partitions
  row_tile_size = num_simd_lanes
  padded_input_size = _align_to(
      input_size,
      math.lcm(num_rows_partitions * row_tile_size, reduce_group_size),
  )
  pad_input_size = padded_input_size - input_size

  # Pad x's ROWS independently of the index count: x's row count is not tied
  # to indices.size (the chunked combine passes the full buffer with a slice of
  # the indices; the truncated-buffer mode passes a packed buffer shorter than
  # the index list). We need (a) row 0..padded index targets valid, and (b) the
  # row count to remain a multiple of the 32-bit packing factor used by the
  # kernel's bitcast view. For the standard x.shape[0] == input_size call this
  # equals the previous ((0, pad_input_size)) padding exactly.
  input_packing = 32 // (dtype_bytes * 8)
  pad_x_rows = max(padded_input_size - x.shape[0], 0)
  pad_x_rows += -(x.shape[0] + pad_x_rows) % input_packing
  x = jnp.pad(
      x,
      ((0, pad_x_rows), (0, aligned_hidden_size - hidden_size)),
      constant_values=0,
  )
  indices = jnp.pad(indices, (0, pad_input_size), constant_values=0)
  topk_weights = jnp.pad(topk_weights, (0, pad_input_size), constant_values=0)
  valid_rows_mask = jnp.pad(valid_rows_mask, (0, pad_input_size), constant_values=False)

  (
      src_indices,
      dst_indices,
      topk_weights,
      num_src_rows_per_row_partition,
      mask,
  ) = _preprocess(
      indices,
      topk_weights,
      valid_rows_mask,
      reduce_group_size,
      num_rows_partitions,
      num_simd_lanes,
  )

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  # Each output row from `main_kernel` will be of type float32, and then
  # casted to the input dtype when doing the filter operation.
  out = pl.kernel(  # pytype: disable=wrong-keyword-args
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_row_partitions=num_rows_partitions,
          num_column_partitions=num_column_partitions,
      ),
      compiler_params=pltpu.CompilerParams(  # pytype: disable=wrong-keyword-args
          **_COMPILER_PARAMS,
      ),
      cost_estimate=get_cost_estimate(
          padded_input_size=padded_input_size,
          aligned_hidden_size=aligned_hidden_size,
          reduce_group_size=reduce_group_size,
          input_dtype_bytes=dtype_bytes,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce",
      **{
          _OUT_KW: jax.ShapeDtypeStruct(
              (padded_input_size // reduce_group_size, aligned_hidden_size),
              jnp.float32,
          ),
          _SCRATCH_KW: dict(  # pylint: disable=use-dict-literal
              num_rows_per_row_partition_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
              prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
              src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
              sem_ref=pltpu.SemaphoreType.DMA((2,)),
          ),
      },
  )(num_src_rows_per_row_partition, x, src_indices, dst_indices, topk_weights)

  # If there is no valid source row in a reduce group, set that group's output
  # to zero.
  return jnp.where(
      mask[:, None],
      out.astype(x.dtype),
      jnp.zeros_like(out, dtype=x.dtype),
  )[: (input_size // reduce_group_size), :hidden_size]


def sc_accumulate_dims(input_size: int, hidden_size: int, reduce_group_size: int = 1):
  """Padded (rows, cols) of the ACCUMULATE buffer for ``ragged_gather_reduce_accumulate``.

  SparseCore-only (reads ``get_tpu_info``). The accumulate buffer must be pre-shaped to these
  padded dims and threaded through the chunk loop so the padding is paid ONCE (not per chunk).
  Mirrors the padding math in ``ragged_gather_reduce`` exactly.
  """
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "sc_accumulate_dims requires SparseCore"
  num_column_partitions = 8
  num_cores = sc_info.num_cores * sc_info.num_subcores
  num_rows_partitions = num_cores // num_column_partitions
  aligned_hidden_size = _align_to(hidden_size, 128 * num_column_partitions)
  padded_input_size = _align_to(
      input_size, math.lcm(num_rows_partitions * sc_info.num_lanes, reduce_group_size)
  )
  return padded_input_size // reduce_group_size, aligned_hidden_size


def ragged_gather_reduce_accumulate(
    accum,
    x,
    indices,
    topk_weights,
    valid_rows_mask,
    reduce_group_size,
    flops_override: int = -1,
    bytes_accessed_override: int = -1,
):
  """In-place accumulate variant of :func:`ragged_gather_reduce` (SparseCore only).

  BLOCKED / NOT WIRED (kept as a validated reference for the eventual infra fix). Correctness and
  in-place aliasing are proven STANDALONE on v5p (chunked accumulate fill == un-chunked
  ``ragged_gather``, max|diff|=0; 0 full-accum-buffer copies; temp flat in n_chunks). BUT it cannot
  be used inside the model's ``lax.scan`` + ``custom_vjp``: initializing the core_map output ref to
  an INPUT value (``jax_core.new_ref(accum)``) makes JAX treat that Ref as a jit output, and
  "mutable array references cannot be returned" across the jit / custom_vjp boundary -- reproduced
  for EVERY wrapper structure tried (jit+donate_argnums, jit without donation, and inline in the
  ambient trace). ``pl.kernel`` avoids the leak only because its output ref comes from
  ``new_ref(lax.empty(...))`` (an internal value, not an input), which cannot be initialized to
  ``accum``; and ``pl.kernel`` exposes no ``input_output_aliases`` (only ``pl.pallas_call`` does, and
  it has no SparseCore subcore mesh). Unblocking needs either ``input_output_aliases`` plumbed
  through ``pl.kernel``, or a native SC scatter-write/accumulate primitive. Until then
  ``chunked_ring_dispatch`` uses the full-buffer ``jnp.where`` merge (correct, cluster-validated,
  N x traffic).

  Identical gather/reduce math to ``ragged_gather_reduce``, but the kernel's output ref is
  INITIALIZED to ``accum`` (an ``sc_accumulate_dims``-shaped float32 buffer), so the kernel writes
  ONLY the valid (this-call's) destination rows in place and leaves every other row at its prior
  ``accum`` value. There is NO final ``where``-zeroing. Threaded across the decouple_dispatch_chunks
  loop this gives
  O(chunk_rows) HBM write + O(chunk_rows) SC compute per chunk (the compaction only processes the
  valid rows), replacing the rung-9 per-chunk full-buffer ``jnp.where`` merge (N x ~14GB traffic).

  Because ``main_kernel`` OVERWRITES ``out[dst]`` (no read-accumulate) and each buffer row is
  written by exactly one chunk (disjoint ``chunk_of_row``), the overwrite-in-place reproduces the
  full-buffer gather bit-exactly. Validated standalone on v5p (num_lanes=8); SC numerics are
  otherwise cluster-gated (CPU has no SparseCore).

  Args:
    accum: float32 ``[sc_accumulate_dims(input_size, hidden)]`` accumulate buffer (donated).
    x, indices, topk_weights, valid_rows_mask, reduce_group_size: as in ``ragged_gather_reduce``.

  Returns:
    The updated accumulate buffer (same padded float32 shape as ``accum``).
  """
  assert x.ndim == 2 and indices.ndim == 1
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "ragged_gather_reduce_accumulate requires SparseCore"

  dtype = x.dtype
  dtype_bytes = jax.dtypes.itemsize_bits(dtype) // 8
  hidden_size = x.shape[-1]
  input_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  num_column_partitions = 8
  assert num_cores % num_column_partitions == 0
  num_rows_partitions = num_cores // num_column_partitions

  aligned_hidden_size = _align_to(hidden_size, 128 * num_column_partitions)
  col_size = aligned_hidden_size // num_column_partitions
  padded_input_size = _align_to(
      input_size, math.lcm(num_rows_partitions * num_simd_lanes, reduce_group_size)
  )
  pad_input_size = padded_input_size - input_size

  input_packing = 32 // (dtype_bytes * 8)
  pad_x_rows = max(padded_input_size - x.shape[0], 0)
  pad_x_rows += -(x.shape[0] + pad_x_rows) % input_packing
  x = jnp.pad(x, ((0, pad_x_rows), (0, aligned_hidden_size - hidden_size)), constant_values=0)
  indices = jnp.pad(indices, (0, pad_input_size), constant_values=0)
  topk_weights = jnp.pad(topk_weights, (0, pad_input_size), constant_values=0)
  valid_rows_mask = jnp.pad(valid_rows_mask, (0, pad_input_size), constant_values=False)

  src_indices, dst_indices, topk_weights, num_src_rows_per_row_partition, _mask = _preprocess(
      indices, topk_weights, valid_rows_mask, reduce_group_size, num_rows_partitions, num_simd_lanes
  )

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  scratch = dict(  # pylint: disable=use-dict-literal
      num_rows_per_row_partition_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
      out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
      prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
      src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
      dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
      topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
      sem_ref=pltpu.SemaphoreType.DMA((2,)),
  )
  body = functools.partial(
      main_kernel,
      core_axis_name=vector_mesh.core_axis_name,
      subcore_axis_name=vector_mesh.subcore_axis_name,
      num_row_partitions=num_rows_partitions,
      num_column_partitions=num_column_partitions,
  )
  cost = get_cost_estimate(
      padded_input_size=padded_input_size,
      aligned_hidden_size=aligned_hidden_size,
      reduce_group_size=reduce_group_size,
      input_dtype_bytes=dtype_bytes,
      flops_override=flops_override,
      bytes_accessed_override=bytes_accessed_override,
  )

  # Mirror pl.kernel's OWN wrapper -- a plain `@jax.jit` around new_ref + core_map that returns
  # `out_ref[...]` (a value) -- EXCEPT the single output ref is initialized to `accum` (NOT lax.empty)
  # so the kernel accumulates in place, leaving unwritten rows at their prior value.
  #
  # The jit boundary is LOAD-BEARING: it fully scopes the new_ref Refs so none escapes. pl.kernel is
  # exactly this shape and composes cleanly with the model's custom_vjp + lax.scan (it already runs
  # inside ring_ragged_unsort's custom_vjp inside the per-layer scan). WITHOUT the jit (inlined in the
  # ambient trace) the Ref leaks out of the enclosing custom_vjp ("mutable array references cannot be
  # returned", rung9c); WITH `donate_argnums` the donated accum Ref leaks across the jit boundary
  # itself (rung9b). So: jit, NO donation. Aliasing (accum buffer -> output, no full-buffer copy)
  # comes from XLA's in-place buffer reuse -- accum is dead after each call (the barrier chain in
  # chunked_ring_dispatch) -- verified on v5p (0 full-accum-buffer copies, temp flat in n_chunks).
  @jax.jit
  def _run(accum, num_src, xp, src_idx, dst_idx, w):
    arg_refs = _tree_util.tree_map(jax_core.new_ref, (num_src, xp, src_idx, dst_idx, w))
    out_ref = jax_core.new_ref(accum)  # init to accum -> in-place accumulate (unwritten rows kept)

    @pl_core.core_map(
        vector_mesh,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(**_COMPILER_PARAMS),
        cost_estimate=cost,
        name="sc_ragged_gather_reduce_accumulate",
    )
    def _(**scratch_kwrefs):
      return body(*arg_refs, out_ref, **scratch_kwrefs)

    return out_ref[...]

  return _run(accum, num_src_rows_per_row_partition, x, src_indices, dst_indices, topk_weights)
