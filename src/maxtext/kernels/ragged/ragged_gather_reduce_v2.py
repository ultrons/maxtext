# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ragged gather reduce kernel implementation from tpu-inference."""
# pylint: disable=line-too-long
# Source from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


# pylint: disable=missing-class-docstring
@dataclasses.dataclass(frozen=True)
class _Config:
  num_row_partitions: int
  num_column_partitions: int
  reduce_group_size: int
  col_size: int
  col_chunk_size: int
  num_row_subchunks: int
  num_simd_lanes: int
  topk_dtype: Any
  in_dtype: Any
  core_axis_name: str
  subcore_axis_name: str

  @property
  def out_pack(self) -> int:
    """Output rows per 32-bit output word: 2 for a bfloat16 output, 1 for float32.

    The output is written through a uint32 view of the HBM buffer, so a
    bfloat16 output packs two consecutive output rows into one word (the even
    row in the low 16 bits, as in the input unpacking below).
    """
    return 32 // jax.dtypes.itemsize_bits(self.in_dtype)

  @property
  def row_chunk_size(self) -> int:
    """Number of rows handled per row-pipeline block."""
    return self.num_simd_lanes * self.num_row_subchunks

  @property
  def row_shift(self) -> int:
    """log2 of how many source rows pack into one uint32 gather element.

    The SparseCore indirect DMA requires 32-bit elements: bfloat16 packs two
    source rows per uint32 (shift 1), float32 is 1:1 (shift 0).
    """
    input_packing = 32 // jax.dtypes.itemsize_bits(self.in_dtype)
    return input_packing.bit_length() - 1


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
    #   write – output rows (same dtype as the input)
    bytes_in = padded_input_size * aligned_hidden_size * input_dtype_bytes  # input rows
    bytes_in += padded_input_size * 4  # src_indices (int32)
    bytes_in += padded_input_size * 4  # dst_indices (int32)
    bytes_in += padded_input_size * 4  # topk_weights (float32)
    output_rows = padded_input_size // reduce_group_size
    bytes_out = output_rows * aligned_hidden_size * input_dtype_bytes  # output rows
    bytes_accessed = bytes_in + bytes_out

  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=bytes_accessed,
      transcendentals=0,
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Inputs:
  num_src_rows_per_row_partition: Any
  x: Any
  indices: Any
  topk_weights: Any
  sorted_by_validity: Any
  empty_words: Any
  num_empty_words_per_row_partition: Any


# pylint: disable=missing-class-docstring
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _Scratch:
  num_rows_per_row_partition_vmem: Any
  prev_iter_last_row_vmem: Any
  prev_iter_last_word_vmem: Any
  prev_dst_row_smem: Any
  sorted_by_validity_vmem: Any
  num_empty_words_vmem: Any
  empty_words_vmem: Any
  zero_row_vmem: Any
  src_indices_vmem: Any
  dst_indices_vmem: Any
  pad_row_vmem: Any
  tw_f32_vmem: Any
  dma_src_row_vmem: Any
  dma_dst_row_vmem: Any
  prev_dst_val_vmem: Any
  out_vmem: Any
  sem: Any

  def __len__(self) -> int:
    return len(dataclasses.fields(self))

  def __getitem__(self, index: Any):
    return getattr(self, dataclasses.fields(self)[index].name)


# pylint: disable=missing-class-docstring
class _CostModelConstants:
  # Limit on the number of outer loop pipeline iterations. Too many iterations
  # cause high cumulative pipeline overhead (e.g., from frequent pipeline
  # startup/teardown bubbles). We try to find partitioning that does not exceed
  # this limit on iterations.
  MAX_ITERATIONS: int = 40

  # Upper cap on the column chunk size processed per inner pipeline step.
  # While larger chunk sizes help utilize bandwidth better, excessively large
  # chunk sizes cause large pipeline bubbles. We cap it here to balance
  # efficiency and bubble sizes.
  MAX_COL_CHUNK_SIZE: int = 1024


# ceil up to the nearest multiple of b.
def _align_to(a, b):
  return pl.cdiv(a, b) * b


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Fallback implementation using JAX ops for non-SparseCore TPU or small inputs."""
  out = x[indices] * topk_weights[:, None].astype(jnp.float32)
  out = jnp.where(valid_rows_mask[:, None], out, 0)
  out = out.reshape(-1, reduce_group_size, out.shape[-1])
  out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
  return out


def _calculate_num_column_partitions(
    hidden_size: int, input_size: int, num_cores: int, num_lanes: int, num_simd_lanes: int
) -> int:
  """Calculates the number of row partitions."""
  # Each column partition should be multiple of 128 (number of lanes) due to
  # DMA requirements.
  # Prefer to use a large number of column partitions, as long as each
  # partition's size is not too small for DMA pipeline efficiency and each
  # partition's size can divide the hidden size.

  # Each column partition will do DMA pipelining on col_size.
  preferred_num_stages = 4
  num_column_partitions = 1
  while (
      num_cores % (num_column_partitions * 2) == 0
      and hidden_size % (num_lanes * num_column_partitions * 2) == 0
      and hidden_size // (num_column_partitions * 2 * num_lanes) >= preferred_num_stages
  ):
    next_candidate = num_column_partitions * 2
    next_row_partitions = num_cores // next_candidate

    # Calculate exactly how many pipeline invocations (outer loop)
    _, row_chunk_size = _calculate_row_tiling(input_size, num_simd_lanes, next_row_partitions)
    num_iterations = input_size // (row_chunk_size * next_row_partitions)

    # Ensure we satisfy the hardware constraint (num_row_partitions <= num_simd_lanes) first.
    if num_cores // num_column_partitions > num_simd_lanes:
      num_column_partitions = next_candidate
      continue

    # Too many iterations cause high cumulative pipeline overhead. Set the
    # limit based on empirical data.
    if num_iterations > _CostModelConstants.MAX_ITERATIONS:
      break

    num_column_partitions = next_candidate

  return num_column_partitions


def _calculate_row_tiling(
    input_size: int,
    num_simd_lanes: int,
    num_row_partitions: int,
) -> tuple[int, int]:
  """Calculates the number of row subchunks and row chunk size."""
  base_block_size = num_simd_lanes * num_row_partitions
  num_row_subchunks = max(1, min(4, pl.cdiv(input_size, base_block_size)))
  row_chunk_size = num_simd_lanes * num_row_subchunks
  return num_row_subchunks, row_chunk_size


def _calculate_col_chunk_size(col_size: int, num_simd_lanes: int) -> int:
  """Picks the column chunk size the inner pipeline gathers at a time.

  The chunk is the largest divisor of ``col_size`` whose gather double-buffer
  still fits comfortably in SparseCore VMEM.
  """
  generation = pltpu.get_tpu_info().generation
  match generation:
    case 6:
      target_bytes = int(256 * 1024 * 0.95)
    case 7:
      target_bytes = int(512 * 1024 * 0.95)
    case _:
      target_bytes = int(128 * 1024 * 0.95)

  # uint32 gather buffer, double-buffered by emit_pipeline.
  bytes_per_col = num_simd_lanes * 4 * 2
  max_safe_col = (target_bytes // bytes_per_col // 128) * 128

  # Larger chunk sizes cause larger pipeline bubbles, so cap it at 1024.
  max_safe_col = min(max_safe_col, _CostModelConstants.MAX_COL_CHUNK_SIZE)

  start_col = (min(col_size, max_safe_col) // 128) * 128
  for chunk in range(start_col, 127, -128):
    if col_size % chunk == 0:
      return chunk
  return 128


def _preprocess(
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    num_row_partitions: int,
    num_simd_lanes: int,
    row_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sorts valid source rows to the front of each row partition.

  Returns:
    sorted_by_validity: original row index of each slot after the stable
      sort, flattened across partitions and padded to ``row_chunk_size``.
    num_src_rows_per_row_partition: valid row count per partition, padded to
      ``num_simd_lanes`` so the kernel can load it as a single vector.
    mask: per output group, whether the group has any valid source row.
  """
  row_partition_size = valid_rows_mask.shape[0] // num_row_partitions
  valid_rows_mask_2d = valid_rows_mask.reshape(num_row_partitions, -1)

  # Stable sort of a boolean key is a stable partition: valid rows keep their
  # relative order and move ahead of the invalid ones.
  sorted_by_validity = jnp.argsort(~valid_rows_mask_2d, descending=False, stable=True, axis=-1)
  sorted_by_validity += jnp.arange(num_row_partitions)[:, None] * row_partition_size

  pad_to = _align_to(row_partition_size, row_chunk_size)
  if pad_to > row_partition_size:
    sorted_by_validity = jnp.pad(
        sorted_by_validity,
        ((0, 0), (0, pad_to - row_partition_size)),
        constant_values=0,
    )
  sorted_by_validity = sorted_by_validity.reshape(-1)

  num_src_rows_per_row_partition = jnp.pad(
      jnp.sum(valid_rows_mask_2d, axis=-1).astype(jnp.int32),
      (0, max(0, num_simd_lanes - num_row_partitions)),
  )
  mask = jnp.any(valid_rows_mask.reshape(-1, reduce_group_size), axis=-1)
  return (
      sorted_by_validity.astype(jnp.int32),
      num_src_rows_per_row_partition,
      mask,
  )


def _pack_scalars_to_vector(scalar_list: list[jax.Array], num_simd_lanes: int) -> jax.Array:
  """Builds a lane vector from per-lane scalars.

  SparseCore cannot store individual scalars into VMEM lanes, so the vector is
  assembled with masked accumulation before being stored.
  """
  idx_vec = jnp.arange(num_simd_lanes)
  vec = jnp.zeros((num_simd_lanes,), jnp.int32)
  for i in range(num_simd_lanes):
    vec += (idx_vec == i).astype(jnp.int32) * scalar_list[i]
  return vec


def _round_f32_bits_to_bf16(bits: jax.Array, is_nan: jax.Array | None = None) -> jax.Array:
  """Rounds float32 bit patterns to bfloat16 bit patterns (uint32 in, low 16 bits out).

  Mirrors the TensorCore ``convert`` that the wrapper's ``astype`` used to run
  on the kernel's float32 output, so the bfloat16 result is bit-identical to
  the old two-pass path: round to nearest even on the magnitude and every NaN
  mapped to the canonical quiet NaN 0x7FC0 (measured on a TPU; see
  ``ragged_gather_reduce_v2_test``). The TensorCore convert also flushes
  denormals to a signed zero; the SparseCore's float32 arithmetic already
  flushes them, so the sums this rounds are never denormal and no flush is
  needed here (checked on a TPU with products down to 2^-136). Runs on the
  SparseCore vector unit and under plain XLA (tests).

  Args:
    bits: float32 values as uint32.
    is_nan: optional precomputed NaN mask (``jnp.isnan`` of the float32
      value, one vector compare in the kernel); derived from ``bits`` when
      omitted.
  """
  lsb = jnp.bitwise_and(jnp.right_shift(bits, 16), jnp.uint32(1))
  rounded = jnp.right_shift(bits + (jnp.uint32(0x7FFF) + lsb), 16)
  if is_nan is None:
    is_nan = jnp.bitwise_and(bits, jnp.uint32(0x7FFFFFFF)) > jnp.uint32(0x7F800000)
  return jnp.where(is_nan, jnp.uint32(0x7FC0), rounded)


def _pack_output_word(prev_word, bf16_bits, dst_row, same_word):
  """Merges one output row's bfloat16 bits into the uint32 word of its row pair.

  ``dst_row`` selects the half (even row: low 16 bits). When the previous row
  belonged to the same pair (``same_word``) the other half is kept, otherwise
  it starts a fresh word, which leaves a missing (empty) partner row at zero.
  ``dst_row`` and ``same_word`` are scalars, so the masks are scalar work and
  the vector unit does one shift, one and, one or per row.
  """
  is_low = jnp.bitwise_and(dst_row, 1) == 0
  shift = jnp.where(is_low, 0, 16)
  keep = jnp.where(is_low, jnp.uint32(0xFFFF0000), jnp.uint32(0x0000FFFF))
  keep = jnp.where(same_word, keep, jnp.uint32(0))
  return jnp.bitwise_or(jnp.bitwise_and(prev_word, keep), jnp.left_shift(bf16_bits, shift))


def _reduce_row(prev_acc, data_f32, same_group, is_pad):
  """One step of the segmented reduction; a padding row carries the previous value unchanged."""
  acc = jnp.where(same_group, prev_acc + data_f32, data_f32)
  return jnp.where(is_pad, prev_acc, acc)


def _plan_row_block(
    dst_rows: list[Any],
    row_valid: list[Any],
    prev_dst: Any,
    *,
    num_simd_lanes: int,
    num_row_subchunks: int,
    out_pack: int,
) -> tuple[list[Any], list[Any], list[list[Any]], list[list[Any]]]:
  """Scalar bookkeeping for one block of ``row_chunk_size`` sorted source rows.

  Args:
    dst_rows: per row, its output row (reduce group index), in sorted order.
    row_valid: per row, whether it lies in the partition's valid prefix.
    prev_dst: effective destination of the row before this block
      (``-out_pack`` before the first block: no group and no output word).

  Returns:
    dst_eff: per row, its effective destination. Rows past the partition's
      valid prefix (padding) become continuation rows: they take the
      destination of the last valid row and, in the reduction, carry its value
      unchanged, so their scatter rewrites the same word with the same value.
      Only the last block has padding, and it always starts with a valid row.
    is_pad: per row, 1 for a padding row, else 0.
    dma_src, dma_dst: per sub-chunk and lane, the VMEM row whose output word
      the lane scatters and the output word it writes. Rows sharing an output
      word within the sub-chunk all scatter the word's last row (idempotent);
      a row whose word continues into a later sub-chunk scatters its own
      partial word, which the later sub-chunk overwrites once this one's DMAs
      have been waited for.
  """
  row_chunk_size = num_simd_lanes * num_row_subchunks
  dst_eff = []
  prev = prev_dst
  for r in range(row_chunk_size):
    dst = jnp.where(row_valid[r], dst_rows[r], prev)
    dst_eff.append(dst)
    prev = dst
  is_pad = [jnp.logical_not(valid).astype(jnp.int32) for valid in row_valid]
  word_eff = [dst // out_pack for dst in dst_eff]

  # For each source row, the VMEM row that will hold its output word's fully
  # assembled value -- the last row of the word's run within this block.
  # Scanning backwards, a row inherits its successor's merge target when they
  # share an output word, otherwise it is its own target.
  merge_target = [None] * row_chunk_size
  for r in reversed(range(row_chunk_size)):
    if r == row_chunk_size - 1:
      merge_target[r] = r
    else:
      same_word_as_next = (word_eff[r] == word_eff[r + 1]).astype(jnp.int32)
      merge_target[r] = same_word_as_next * merge_target[r + 1] + (1 - same_word_as_next) * r

  dma_src = []
  dma_dst = []
  for s in range(num_row_subchunks):
    sub_src = []
    sub_dst = []
    for i in range(num_simd_lanes):
      r = s * num_simd_lanes + i
      is_final_write = merge_target[r] < (s + 1) * num_simd_lanes
      sub_src.append(jnp.where(is_final_write, merge_target[r] - s * num_simd_lanes, i))
      sub_dst.append(word_eff[r])
    dma_src.append(sub_src)
    dma_dst.append(sub_dst)
  return dst_eff, is_pad, dma_src, dma_dst


def _empty_output_words(
    mask: jax.Array,
    num_row_partitions: int,
    out_pack: int,
    num_simd_lanes: int,
) -> tuple[jax.Array, jax.Array]:
  """Lists, per row partition, the output words that no valid source row writes.

  The kernel only scatters words that own at least one valid source row, so it
  zero-fills these itself (the wrapper used to mask them on the TensorCore).

  Returns:
    empty_words: int32 ``(num_row_partitions * pad_to,)``: per partition, the
      global indices of its empty output words, compacted to the front. Slots
      past the count repeat the partition's first empty word, so the kernel's
      last, partial block of zero writes only re-zeroes an empty word.
    num_empty_words: int32 per partition, padded to ``num_simd_lanes``.
  """
  num_words = mask.shape[0] // out_pack
  word_valid = jnp.any(mask.reshape(num_words, out_pack), axis=-1).reshape(num_row_partitions, -1)
  words_per_partition = word_valid.shape[1]
  pad_to = _align_to(words_per_partition, num_simd_lanes)
  # A stable sort of the validity bit puts the empty words first, in index order.
  empty_first = jnp.argsort(word_valid, axis=-1, stable=True).astype(jnp.int32)
  num_empty = jnp.sum(~word_valid, axis=-1).astype(jnp.int32)
  empty_first = jnp.pad(empty_first, ((0, 0), (0, pad_to - words_per_partition)))
  slot = jnp.arange(pad_to, dtype=jnp.int32)[None, :]
  empty_first = jnp.where(slot < num_empty[:, None], empty_first, empty_first[:, :1])
  empty_first = empty_first + (jnp.arange(num_row_partitions, dtype=jnp.int32) * words_per_partition)[:, None]
  num_empty = jnp.pad(num_empty, (0, max(0, num_simd_lanes - num_row_partitions)))
  return empty_first.reshape(-1), num_empty


def _row_gather_spec(
    sorted_by_validity_vmem: jax.Ref,
    sub: int,
    *,
    num_simd_lanes: int,
    row_chunk_size: int,
) -> pl.BlockSpec:
  """Indirect BlockSpec gathering sub-chunk ``sub``'s rows of a 1-D input."""
  return pl.BlockSpec(
      (pl.Indirect(num_simd_lanes),),
      lambda i, s=sub: (sorted_by_validity_vmem[pl.ds(i * row_chunk_size + s * num_simd_lanes, num_simd_lanes)],),
  )


def main_kernel(
    inputs: _Inputs,
    out_hbm_ref: jax.Ref,
    scratch: _Scratch,
    *,
    cfg: _Config,
):
  """Main kernel for ragged gather-reduce."""
  # Step 1: Resolve this core's row/column partition and its column slice.
  num_simd_lanes = cfg.num_simd_lanes
  col_chunk_size = cfg.col_chunk_size
  num_row_subchunks = cfg.num_row_subchunks
  row_chunk_size = cfg.row_chunk_size
  out_pack = cfg.out_pack

  num_col_chunks = cfg.col_size // col_chunk_size

  core_id = jax.lax.axis_index((cfg.core_axis_name, cfg.subcore_axis_name))
  row_partition_id = core_id // cfg.num_column_partitions
  col_partition_id = core_id % cfg.num_column_partitions

  row_partition_size_padded = inputs.sorted_by_validity.shape[0] // cfg.num_row_partitions
  row_start_padded = row_partition_id * row_partition_size_padded
  col_start = col_partition_id * cfg.col_size
  words_per_partition_padded = inputs.empty_words.shape[0] // cfg.num_row_partitions
  words_start_padded = row_partition_id * words_per_partition_padded

  # Step 2: Stage this partition's row count, sort permutation and empty-word
  # list into VMEM.
  recv_sem = scratch.sem.at[0]
  send_sem = scratch.sem.at[1]
  staging_dmas = (
      pltpu.make_async_copy(
          inputs.num_src_rows_per_row_partition.at[pl.ds(0, num_simd_lanes)],
          scratch.num_rows_per_row_partition_vmem,
          recv_sem,
      ),
      pltpu.make_async_copy(
          inputs.sorted_by_validity.at[pl.ds(row_start_padded, row_partition_size_padded)],
          scratch.sorted_by_validity_vmem,
          recv_sem,
      ),
      pltpu.make_async_copy(
          inputs.num_empty_words_per_row_partition.at[pl.ds(0, num_simd_lanes)],
          scratch.num_empty_words_vmem,
          recv_sem,
      ),
      pltpu.make_async_copy(
          inputs.empty_words.at[pl.ds(words_start_padded, words_per_partition_padded)],
          scratch.empty_words_vmem,
          recv_sem,
      ),
  )
  for dma in staging_dmas:
    dma.start()
  for dma in staging_dmas:
    dma.wait()

  num_rows_per_row_partition = scratch.num_rows_per_row_partition_vmem[...]
  num_empty_words_per_row_partition = scratch.num_empty_words_vmem[...]
  num_rows_current_row_partition = jnp.array(0, jnp.int32)
  num_empty_words_current_row_partition = jnp.array(0, jnp.int32)
  for i in range(cfg.num_row_partitions):
    num_rows_current_row_partition = jnp.where(
        row_partition_id == i,
        num_rows_per_row_partition[i],
        num_rows_current_row_partition,
    )
    num_empty_words_current_row_partition = jnp.where(
        row_partition_id == i,
        num_empty_words_per_row_partition[i],
        num_empty_words_current_row_partition,
    )
  num_row_blocks = pl.cdiv(num_rows_current_row_partition, row_chunk_size)
  num_zero_blocks = pl.cdiv(num_empty_words_current_row_partition, num_simd_lanes)

  # The output is written through a 32-bit view: a bfloat16 output packs two
  # consecutive output rows per uint32 word (``cfg.out_pack``), float32 is 1:1.
  # Single-row DMAs into a 16-bit-tiled buffer are not possible, so the kernel
  # assembles each pair of output rows into one word and scatters whole words.
  out_words_hbm_ref = out_hbm_ref.bitcast(jnp.uint32)

  # Step 3: Zero-fill the output words that own no valid source row. The
  # gather pipeline below never visits them, and the wrapper no longer masks.
  def zero_row_loop(col_offset):
    scratch.zero_row_vmem[pl.ds(col_offset, num_simd_lanes)] = jnp.zeros((num_simd_lanes,), jnp.uint32)

  plsc.parallel_loop(0, cfg.col_size, step=num_simd_lanes)(zero_row_loop)

  @pl.loop(0, num_zero_blocks)
  def zero_fill_loop(block):
    empty_word_slice = scratch.empty_words_vmem[pl.ds(block * num_simd_lanes, num_simd_lanes)]
    copies = []
    for i in range(num_simd_lanes):
      copy = pltpu.make_async_copy(
          scratch.zero_row_vmem.at[pl.ds(0, cfg.col_size)],
          out_words_hbm_ref.at[empty_word_slice[i], pl.ds(col_start, cfg.col_size)],
          send_sem,
      )
      copy.start()
      copies.append(copy)
    for copy in copies:
      copy.wait()

  # Step 4: Run the gather / weighted segmented-reduce / scatter pipeline.

  # The SparseCore indirect DMA requires 32-bit elements, so x is gathered
  # through a uint32 reinterpretation. bfloat16 packs two source rows per
  # uint32 row (row index >> 1); float32 is 1:1 (row index unchanged).
  in_32b_hbm_ref = inputs.x.bitcast(jnp.uint32)

  # Sentinel for the cross-block reduction carry (no previous group, and no
  # previous output word under either rounding of the division by out_pack).
  scratch.prev_dst_row_smem[0] = -out_pack

  # One gather per sub-chunk for ``indices``, then the same for
  # ``topk_weights``.
  row_pipeline_in_specs = (
      tuple(
          _row_gather_spec(
              scratch.sorted_by_validity_vmem,
              sub,
              num_simd_lanes=num_simd_lanes,
              row_chunk_size=row_chunk_size,
          )
          for sub in range(num_row_subchunks)
      )
      * 2
  )

  @functools.partial(
      pltpu.emit_pipeline,
      grid=(num_row_blocks,),
      in_specs=row_pipeline_in_specs,
      out_specs=(),
  )
  def row_pipeline(*args):
    src_indices_refs = args[:num_row_subchunks]
    topk_weights_refs = args[num_row_subchunks : 2 * num_row_subchunks]
    # pylint: disable=unbalanced-tuple-unpacking
    (
        src_indices_vmem_sc,
        dst_indices_vmem_sc,
        pad_row_vmem_sc,
        tw_f32_vmem_sc,
        dma_src_row_vmem_sc,
        dma_dst_row_vmem_sc,
        prev_dst_val_vmem_sc,
        out_vmem_sc,
        sem_sc,
    ) = args[-9:]

    row_block_id = pl.program_id(0)

    # Destination output row of each source row in this block, as sorted.
    dst_indices_list = [
        scratch.sorted_by_validity_vmem[
            pl.ds(
                row_block_id * row_chunk_size + s * num_simd_lanes,
                num_simd_lanes,
            )
        ]
        // cfg.reduce_group_size
        for s in range(num_row_subchunks)
    ]

    dst_rows = [dst_indices_list[r // num_simd_lanes][r % num_simd_lanes] for r in range(row_chunk_size)]
    row_valid = [(row_block_id * row_chunk_size + r) < num_rows_current_row_partition for r in range(row_chunk_size)]
    dst_eff, is_pad, dma_src_rows, dma_dst_rows = _plan_row_block(
        dst_rows,
        row_valid,
        scratch.prev_dst_row_smem[0],
        num_simd_lanes=num_simd_lanes,
        num_row_subchunks=num_row_subchunks,
        out_pack=out_pack,
    )

    # Stage the gathered indices/weights, the effective destinations and the
    # padding flags in VMEM.
    for s in range(num_row_subchunks):
      sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
      lanes = slice(s * num_simd_lanes, (s + 1) * num_simd_lanes)
      src_indices_vmem_sc[sub] = src_indices_refs[s][...]
      dst_indices_vmem_sc[sub] = _pack_scalars_to_vector(dst_eff[lanes], num_simd_lanes)
      pad_row_vmem_sc[sub] = _pack_scalars_to_vector(is_pad[lanes], num_simd_lanes)

      tw = topk_weights_refs[s][...]
      if cfg.topk_dtype == jnp.bfloat16:
        tw_f32 = plsc.bitcast(jnp.bitwise_left_shift(tw, 16), jnp.float32)
      else:
        tw_f32 = plsc.bitcast(tw, jnp.float32)
      tw_f32_vmem_sc[sub] = tw_f32

    # For each sub-chunk, the destination of the row just before it -- the
    # seed for the segmented reduction's "same group as previous row" test.
    for s in range(num_row_subchunks):
      if s == 0:
        prev_dst = scratch.prev_dst_row_smem[0]
      else:
        prev_dst = dst_eff[s * num_simd_lanes - 1]
      prev_dst_val_vmem_sc[pl.ds(s * num_simd_lanes, num_simd_lanes)] = jnp.broadcast_to(prev_dst, (num_simd_lanes,))

    for s in range(num_row_subchunks):
      sub = pl.ds(s * num_simd_lanes, num_simd_lanes)
      dma_src_row_vmem_sc[sub] = _pack_scalars_to_vector(dma_src_rows[s], num_simd_lanes)
      dma_dst_row_vmem_sc[sub] = _pack_scalars_to_vector(dma_dst_rows[s], num_simd_lanes)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(num_row_subchunks, num_col_chunks),
        in_specs=pl.BlockSpec(
            (pl.Indirect(num_simd_lanes), col_chunk_size),
            lambda s, c: (
                jnp.bitwise_right_shift(
                    src_indices_vmem_sc[pl.ds(s * num_simd_lanes, num_simd_lanes)],
                    cfg.row_shift,
                ),
                col_start // col_chunk_size + c,
            ),
        ),
        out_specs=(),
    )
    def col_pipeline(gather_ref, sem_inner):
      s = pl.program_id(0)
      c = pl.program_id(1)
      col_hbm_start = col_start + c * col_chunk_size
      send_sem_inner = sem_inner.at[1]

      row_slice = pl.ds(s * num_simd_lanes, num_simd_lanes)
      tw_slice = tw_f32_vmem_sc[row_slice]
      dst_slice = dst_indices_vmem_sc[row_slice]
      pad_slice = pad_row_vmem_sc[row_slice]
      src_idx_slice = src_indices_vmem_sc[row_slice]
      prev_dst_vals_vec = prev_dst_val_vmem_sc[row_slice]

      def col_loop(col_compute_offset):
        col_slice = pl.ds(col_compute_offset, num_simd_lanes)
        # Running sum and running output word, seeded by the carry from the
        # previous sub-chunk.
        previous_accumulated_data = scratch.prev_iter_last_row_vmem[c, col_slice]
        previous_word = scratch.prev_iter_last_word_vmem[c, col_slice]

        for row_src in range(num_simd_lanes):
          val_u32 = gather_ref[row_src, col_slice]
          if cfg.in_dtype == jnp.bfloat16:
            # The two bfloat16 rows packed in one uint32 word sit in the low
            # (even row) or high (odd row) 16 bits. Shift the wanted half
            # into the float32 sign/exponent position and clear the rest.
            shift = jnp.where(jnp.bitwise_and(src_idx_slice[row_src], 1) == 0, 16, 0)
            shifted = jnp.bitwise_and(jnp.left_shift(val_u32, shift), jnp.uint32(0xFFFF0000))
            data_f32 = plsc.bitcast(shifted, jnp.float32)
          else:
            data_f32 = plsc.bitcast(val_u32, jnp.float32)
          data_f32 *= tw_slice[row_src]

          # Reduction: accumulate while the destination group is unchanged,
          # restart otherwise. Sorting guarantees rows of one group are
          # contiguous. Padding rows carry the previous value unchanged.
          dst_row_hbm = dst_slice[row_src]
          if row_src == 0:
            prev_dst = prev_dst_vals_vec[0]
          else:
            prev_dst = dst_slice[row_src - 1]
          accumulated_data = _reduce_row(
              previous_accumulated_data,
              data_f32,
              dst_row_hbm == prev_dst,
              pad_slice[row_src] != 0,
          )
          previous_accumulated_data = accumulated_data

          # Round the running float32 sum to the output dtype and place it in
          # its half of the output word. The last row of a group leaves the
          # group's final rounded value in the word; the word is complete at
          # the last row of its pair of groups.
          if out_pack == 1:
            word = plsc.bitcast(accumulated_data, jnp.uint32)
          else:
            bf16_bits = _round_f32_bits_to_bf16(plsc.bitcast(accumulated_data, jnp.uint32), jnp.isnan(accumulated_data))
            word = _pack_output_word(
                previous_word,
                bf16_bits,
                dst_row_hbm,
                dst_row_hbm // out_pack == prev_dst // out_pack,
            )
          previous_word = word

          out_vmem_sc[row_src, col_slice] = word
          if row_src == num_simd_lanes - 1:
            scratch.prev_iter_last_row_vmem[c, col_slice] = accumulated_data
            scratch.prev_iter_last_word_vmem[c, col_slice] = word

      plsc.parallel_loop(0, col_chunk_size, step=num_simd_lanes)(col_loop)

      # Scatter every source row's output word to its destination word. Rows
      # that share a word write the same value (idempotent).
      dma_src_row_slice = dma_src_row_vmem_sc[row_slice]
      dma_dst_row_slice = dma_dst_row_vmem_sc[row_slice]
      copies = []
      for i in range(num_simd_lanes):
        copy = pltpu.make_async_copy(
            out_vmem_sc.at[dma_src_row_slice[i], pl.ds(0, col_chunk_size)],
            out_words_hbm_ref.at[dma_dst_row_slice[i], pl.ds(col_hbm_start, col_chunk_size)],
            send_sem_inner,
        )
        copy.start()
        copies.append(copy)
      for copy in copies:
        copy.wait()

    # pylint: disable=no-value-for-parameter
    col_pipeline(in_32b_hbm_ref, scratches=(sem_sc,))
    scratch.prev_dst_row_smem[0] = dst_eff[-1]

  row_pipeline(
      *([inputs.indices] * num_row_subchunks),
      *([inputs.topk_weights] * num_row_subchunks),
      scratches=(
          scratch.src_indices_vmem,
          scratch.dst_indices_vmem,
          scratch.pad_row_vmem,
          scratch.tw_f32_vmem,
          scratch.dma_src_row_vmem,
          scratch.dma_dst_row_vmem,
          scratch.prev_dst_val_vmem,
          scratch.out_vmem,
          scratch.sem,
      ),
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        "reduce_group_size",
        "enforce_fallback",
        "flops_override",
        "bytes_accessed_override",
        "use_single_sparsecore",
    ),
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
    use_single_sparsecore: bool = False,
) -> jax.Array:
  """Gathers ``x`` by ``indices``, weights and masks, then reduces by group.

  Args:
    x: 2-D input features, ``(num_rows, hidden_size)``.
    indices: 1-D gather indices, ``(input_size,)``.
    topk_weights: 1-D per-row weights, ``(input_size,)``.
    valid_rows_mask: 1-D bool mask of valid gathered rows, ``(input_size,)``.
    reduce_group_size: number of consecutive rows summed into one output row.

  Returns:
    Reduced output, ``(input_size // reduce_group_size, hidden_size)``.
  """
  # Step 1: Choose the implementation (TensorCore fallback or SparseCore).
  # Guard against eager initialization on non-TPU hardware (e.g. during CPU tests).
  # pltpu.get_tpu_info() expects TPU hardware and will crash if executed on CPU.
  if enforce_fallback or not pltpu.is_tpu_device():
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  # For a small {input + output} both likely fit in TensorCore VMEM, where a
  # plain TC gather-reduce beats routing through SparseCore and HBM. This
  # also keeps the kernel off configs with num_row_partitions > num_simd_lanes.
  dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
  # if (jnp.size(x) * dtype_bytes * 2
  #         < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6):
  #     return _fallback_implementation(x, indices, topk_weights,
  #                                     valid_rows_mask, reduce_group_size)

  # Step 2: Derive the kernel configuration (core grid and column tiling).
  hidden_size = x.shape[-1]
  input_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_lanes = pltpu.get_tpu_info().num_lanes
  num_sc_cores = 1 if use_single_sparsecore else sc_info.num_cores
  num_cores = num_sc_cores * sc_info.num_subcores

  num_column_partitions = _calculate_num_column_partitions(hidden_size, input_size, num_cores, num_lanes, num_simd_lanes)
  num_row_partitions = num_cores // num_column_partitions
  assert num_row_partitions <= num_simd_lanes, f"{num_row_partitions=} must be <= {num_simd_lanes=}"
  num_row_subchunks, row_chunk_size = _calculate_row_tiling(input_size, num_simd_lanes, num_row_partitions)

  aligned_hidden_size = _align_to(hidden_size, 128 * num_column_partitions)
  col_size = aligned_hidden_size // num_column_partitions
  col_chunk_size = _calculate_col_chunk_size(col_size, num_simd_lanes)

  # Step 3: Pre-process inputs (weights, padding, sort by validity).
  # The kernel gathers x through a uint32 reinterpretation; carry the weights
  # the same way so they can be bitcast back to float32 on SparseCore.
  if topk_weights.dtype == jnp.bfloat16:
    topk_weights_u32 = jax.lax.bitcast_convert_type(topk_weights, jnp.uint16).astype(jnp.uint32)
  else:
    topk_weights_u32 = jax.lax.bitcast_convert_type(topk_weights, jnp.uint32)

  # Pad the input so each row partition holds a whole number of output words
  # (``out_pack`` reduce groups each); no group, and no pair of output rows
  # packed into one 32-bit word, is then split across two physical cores.
  out_pack = 32 // jax.dtypes.itemsize_bits(x.dtype)
  padded_input_size = _align_to(input_size, num_row_partitions * reduce_group_size * out_pack)
  valid_rows_mask = jnp.pad(
      valid_rows_mask,
      (0, padded_input_size - input_size),
      constant_values=False,
  )

  sorted_by_validity, num_src_rows_per_row_partition, mask = _preprocess(
      valid_rows_mask,
      reduce_group_size,
      num_row_partitions,
      num_simd_lanes,
      row_chunk_size,
  )
  empty_words, num_empty_words_per_row_partition = _empty_output_words(mask, num_row_partitions, out_pack, num_simd_lanes)

  # Step 4: Launch the SparseCore kernel.
  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=num_sc_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )

  cfg = _Config(
      num_row_partitions=num_row_partitions,
      num_column_partitions=num_column_partitions,
      reduce_group_size=reduce_group_size,
      col_size=col_size,
      col_chunk_size=col_chunk_size,
      num_row_subchunks=num_row_subchunks,
      num_simd_lanes=num_simd_lanes,
      topk_dtype=topk_weights.dtype,
      in_dtype=x.dtype,
      core_axis_name=vector_mesh.core_axis_name,
      subcore_axis_name=vector_mesh.subcore_axis_name,
  )

  # The kernel writes the final output dtype itself (rounded and packed on
  # the SparseCore) and zero-fills the output rows of empty groups, so no
  # TensorCore pass over the output is needed afterwards.
  out = pl.kernel(
      functools.partial(main_kernel, cfg=cfg),
      out_type=jax.ShapeDtypeStruct(
          (padded_input_size // reduce_group_size, aligned_hidden_size),
          x.dtype,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
          needs_layout_passes=False,
      ),
      cost_estimate=get_cost_estimate(
          padded_input_size=padded_input_size,
          aligned_hidden_size=aligned_hidden_size,
          reduce_group_size=reduce_group_size,
          input_dtype_bytes=dtype_bytes,
          flops_override=flops_override,
          bytes_accessed_override=bytes_accessed_override,
      ),
      scratch_types=(  # pyrefly: ignore[bad-argument-type]
          _Scratch(
              num_rows_per_row_partition_vmem=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              prev_iter_last_row_vmem=pltpu.VMEM((col_size // col_chunk_size, col_chunk_size), jnp.float32),
              prev_iter_last_word_vmem=pltpu.VMEM((col_size // col_chunk_size, col_chunk_size), jnp.uint32),
              prev_dst_row_smem=pltpu.SMEM((1,), jnp.int32),
              sorted_by_validity_vmem=pltpu.VMEM((sorted_by_validity.size // num_row_partitions,), jnp.int32),
              num_empty_words_vmem=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              empty_words_vmem=pltpu.VMEM((empty_words.size // num_row_partitions,), jnp.int32),
              zero_row_vmem=pltpu.VMEM((col_size,), jnp.uint32),
              src_indices_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              dst_indices_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              pad_row_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              tw_f32_vmem=pltpu.VMEM((row_chunk_size,), jnp.float32),
              dma_src_row_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              dma_dst_row_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              prev_dst_val_vmem=pltpu.VMEM((row_chunk_size,), jnp.int32),
              out_vmem=pltpu.VMEM((num_simd_lanes, col_chunk_size), jnp.uint32),
              sem=pltpu.SemaphoreType.DMA((2,)),
          ),
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce_v2",
  )(
      _Inputs(
          num_src_rows_per_row_partition=num_src_rows_per_row_partition,
          x=x,
          indices=indices,
          topk_weights=topk_weights_u32,
          sorted_by_validity=sorted_by_validity,
          empty_words=empty_words,
          num_empty_words_per_row_partition=num_empty_words_per_row_partition,
      ),
  )

  # Step 5: Drop the padding rows/columns, if any (a no-op slice in the
  # production shapes, where the kernel output is consumed as is).
  num_output_rows = input_size // reduce_group_size
  if out.shape != (num_output_rows, hidden_size):
    out = out[:num_output_rows, :hidden_size]
  return out
