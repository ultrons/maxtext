"""Block-scaled fp8 tgmm (weight-grad GEMM): dW = x_sorted^T @ dout with BOTH operands e4m3 and
per-gm-segment dynamic scales along the contracting m dim.

NVIDIA's MLPerf DSv3 recipe runs the wgrad GEMM in fp8 with fine (1x32) blocks along the
contracting dim (arXiv:2506.08027: e5m2 grads degrade under block scaling; all-e4m3 matches bf16).
The TPU mapping here uses the tokamax tgmm_v2 group-major (gm) tiles as the quantization blocks:
each gm tile is a group-aligned dynamic segment of m rows, so every kernel step has EXACTLY ONE
static scale row per operand (BlockSpec indexed by gm_id) and the dot runs on raw e4m3 with the
scale outer-product applied post-dot per step (VPU) -- no in-loop dequant, so the MXU fast-path
block-size constraint does not apply and short/ragged segments are fine.

Structure vendored from tokamax pallas_mosaic_tpu_v2_tgmm_kernel.py (helpers imported from the
installed package; only the inner kernel, block specs, kernel main, and entry are redefined).
The caller-side segment quantize replicates the kernel's own gm decomposition (fill_metadata):
per group g with running start offset s: local = s % size_lhs_sublane; the group takes
ceil((group_size + local) / tile_m) tiles; tile 0 spans min(tile_m - local, remaining) rows and
subsequent tiles are tile_m-row (sublane-aligned) -- property-tested bit-for-bit in
tests (test_tgmm_block_meta) against an independent numpy transcription.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_lib

E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Caller-side gm metadata (jnp transcription of gmm_v2.fill_metadata)
# ---------------------------------------------------------------------------
def compute_gm_metadata(group_sizes, tile_m, sublane, max_num_gm):
  """Returns (gm_m_offsets [max_num_gm+1], num_gm) matching the kernel's fill_metadata.

  gm_m_offsets[i] is the starting m row of gm tile i; entries beyond num_gm are filled with the
  total m so searchsorted-based segment ids saturate on the last real segment.
  """
  size_group = group_sizes.shape[0]
  total_m = jnp.sum(group_sizes)

  def outer(g, carry):
    num_gm, start, offs = carry
    gsz = group_sizes[g]
    end = start + gsz
    local = start % sublane
    n_tiles = jnp.where(gsz > 0, pl.cdiv(gsz + local, tile_m), 0)

    def inner(t, c):
      cur, offs_ = c
      loc = cur % sublane
      tm = jnp.minimum(tile_m - loc, end - cur)
      offs_ = offs_.at[t].set(cur)
      offs_ = offs_.at[t + 1].set(cur + tm)
      return cur + tm, offs_

    _, offs = lax.fori_loop(num_gm, num_gm + n_tiles, inner, (start, offs))
    return num_gm + n_tiles, end, offs

  offs0 = jnp.zeros((max_num_gm + 1,), jnp.int32)
  num_gm, _, offs = lax.fori_loop(0, size_group, outer, (jnp.int32(0), jnp.int32(0), offs0))
  # saturate unused tail entries at total_m so segment ids clamp to the last real segment
  idx = jnp.arange(max_num_gm + 1)
  offs = jnp.where(idx > num_gm, total_m.astype(jnp.int32), offs)
  return offs, num_gm


def segment_quantize(a, gm_m_offsets, max_num_gm):
  """Per-gm-segment e4m3 quantize of a [m, c] operand.

  Returns (q e4m3 [m, c], scales f32 [max_num_gm, c]); scale row i = amax over segment i's rows
  / 448 (+eps). Rows beyond the last segment get segment id num_gm-1 (saturated offsets) --
  they are masked in-kernel anyway.
  """
  m = a.shape[0]
  seg_id = jnp.clip(
      jnp.searchsorted(gm_m_offsets[1:], jnp.arange(m, dtype=jnp.int32), side="right"),
      0, max_num_gm - 1,
  )
  af = jnp.abs(a.astype(jnp.float32))
  amax = jax.ops.segment_max(af, seg_id, num_segments=max_num_gm)  # [max_num_gm, c]
  amax = jnp.where(jnp.isfinite(amax), amax, 0.0)  # empty segments give -inf
  scales = amax / E4M3_MAX + 1e-30
  q = jnp.clip(a.astype(jnp.float32) / scales[seg_id], -E4M3_MAX, E4M3_MAX).astype(
      jnp.float8_e4m3fn
  )
  return q, scales


# ---------------------------------------------------------------------------
# Kernel (deltas vs tgmm_lib.tgmm_inner_kernel: per-step per-gm scale multiply)
# ---------------------------------------------------------------------------
def _tile_partial(lhs_tile, rhs_tile, ls, rs, m_start_local, m_end_local):
  """Pure per-tile math (unit-testable outside Pallas): mask rows to [m_start_local,
  m_end_local), raw e4m3 dot in f32, then the per-gm-segment scale outer product."""
  lhs_iota = lax.broadcasted_iota(jnp.int32, lhs_tile.shape, 0)
  lhs_masked = jnp.where(
      jnp.logical_and(m_start_local <= lhs_iota, lhs_iota < m_end_local), lhs_tile, 0
  )
  rhs_iota = lax.broadcasted_iota(jnp.int32, rhs_tile.shape, 0)
  rhs_masked = jnp.where(
      jnp.logical_and(m_start_local <= rhs_iota, rhs_iota < m_end_local), rhs_tile, 0
  )
  partial = jax.lax.dot_general(
      lhs_masked, rhs_masked, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
  )
  return partial * ls.reshape(-1, 1) * rs.reshape(1, -1)


def _inner_kernel(
    tiled_lhs_ref,        # [tile_m//sublane, sublane, tile_k] e4m3
    tiled_lhs_scale_ref,  # [1, 1, tile_k] f32
    tiled_rhs_ref,        # [tile_m//sublane, sublane, tile_n] e4m3
    tiled_rhs_scale_ref,  # [1, 1, tile_n] f32
    tiled_out_ref,        # [None, tile_k, tile_n]
    acc_ref,              # [tile_k, tile_n] f32
    metadata_ref,
    *,
    cfgs,
):
  tiled_lhs_ref = tiled_lhs_ref.reshape(-1, tiled_lhs_ref.shape[-1])
  tiled_rhs_ref = tiled_rhs_ref.reshape(-1, tiled_rhs_ref.shape[-1])
  gm_id = pl.program_id(2)

  def _matmul(is_new_group: bool, is_group_changing: bool):
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset
    partial = _tile_partial(
        tiled_lhs_ref[...],
        tiled_rhs_ref[...],
        tiled_lhs_scale_ref[0, 0],
        tiled_rhs_scale_ref[0, 0],
        m_start_local,
        m_end_local,
    )

    acc = partial
    if not is_new_group:
      acc += acc_ref[...]
    if is_group_changing:
      tiled_out_ref[...] = acc.astype(tiled_out_ref.dtype)
    else:
      acc_ref[...] = acc

  prev_gm_id = jnp.where(gm_id > 0, gm_id - 1, 0)
  is_first_gm = gm_id == 0
  group_id_changed = (
      metadata_ref.gm_id_to_group_id[gm_id] != metadata_ref.gm_id_to_group_id[prev_gm_id]
  )
  new_group = jnp.logical_or(is_first_gm, group_id_changed)
  is_last_gm = gm_id == (pl.num_programs(2) - 1)
  next_gm_id = jnp.where(is_last_gm, gm_id, gm_id + 1)
  group_is_changing = jnp.logical_or(
      is_last_gm,
      metadata_ref.gm_id_to_group_id[gm_id] != metadata_ref.gm_id_to_group_id[next_gm_id],
  )
  lax.cond(
      new_group,
      lambda: lax.cond(
          group_is_changing,
          lambda: _matmul(True, True),
          lambda: _matmul(True, False),
      ),
      lambda: lax.cond(
          group_is_changing,
          lambda: _matmul(False, True),
          lambda: _matmul(False, False),
      ),
  )


def _block_specs(metadata_ref, cfgs):
  index_map = tgmm_lib.TgmmIndexMaps(metadata_ref, cfgs)
  bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)
  lhs_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      index_map.lhs_index_map,
  )
  rhs_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      index_map.rhs_index_map,
  )
  # per-gm scale rows: [max_num_gm, K/N] sliced (gm_id, k_id/n_id)
  lhs_scale_spec = pl.BlockSpec(
      (1, 1, cfgs.tiles.tile_k), lambda n_id, k_id, gm_id: (gm_id, 0, k_id)
  )
  rhs_scale_spec = pl.BlockSpec(
      (1, 1, cfgs.tiles.tile_n), lambda n_id, k_id, gm_id: (gm_id, 0, n_id)
  )
  out_spec = pl.BlockSpec(
      (None, cfgs.tiles.tile_k, cfgs.tiles.tile_n), index_map.out_index_map
  )
  return (lhs_spec, lhs_scale_spec, rhs_spec, rhs_scale_spec), out_spec


def _kernel_main(
    lhs_group_sizes_ref,
    group_offset_ref,
    lhs_ref,        # [m//sublane, sublane, k] e4m3
    lhs_scale_ref,  # [max_num_gm, aligned_k] f32
    rhs_ref,        # [m//sublane, sublane, n] e4m3
    rhs_scale_ref,  # [max_num_gm, aligned_n] f32
    out_ref,
    acc_ref,
    metadata_ref,
    zero_ref,
    semaphore_ref,
    *,
    cfgs,
    skip_zero_out=False,
):
  # skip_zero_out (interpret-mode only): tokamax's zero_out_start reshapes a VMEM scratch ref,
  # unsupported by interpret-mode discharge; the entry post-masks empty groups instead.
  if not skip_zero_out:
    num_groups_to_zero = tgmm_lib.zero_out_start(
        lhs_group_sizes_ref, group_offset_ref, out_ref, zero_ref, semaphore_ref
    )
  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.dims.size_n, cfgs.tiles.tile_n)
  num_gm = gmm_v2.fill_metadata(
      lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs
  )
  in_specs, out_spec = _block_specs(metadata_ref, cfgs)
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(_inner_kernel, cfgs=cfgs),
      grid=(num_n, num_k, num_gm),
      in_specs=in_specs,
      out_specs=out_spec,
  )
  # lhs/rhs arrive PRE-reshaped to [m//sublane, sublane, c] (done in the entry: the in-kernel HBM
  # ref reshape tokamax uses is not supported by interpret-mode state discharge; the pre-reshape is
  # the same row-major view).
  pipeline_fn(
      lhs_ref, lhs_scale_ref, rhs_ref, rhs_scale_ref, out_ref,
      scratches=[acc_ref, metadata_ref],
  )
  if not skip_zero_out:
    tgmm_lib.zero_out_end(num_groups_to_zero, out_ref, semaphore_ref)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def tgmm_block_fp8(
    lhs,  # [m, k] bf16 (x_sorted)
    rhs,  # [m, n] bf16/f32 (dout)
    group_sizes,
    num_actual_groups,
    group_offset=None,
    *,
    quantize_lhs=True,
    tile_info=tgmm_lib.calculate_tgmm_tiling,
    vmem_limit_bytes=None,
    preferred_element_type=None,
    acc_dtype=None,
    interpret=False,
):
  """Drop-in for tokamax tgmm_v2 with gm-segment block-scaled e4m3 operands.

  quantize_lhs=False keeps the lhs exact... (not supported yet: the kernel expects both scale
  operands; pass an all-ones scale + a bf16->e4m3-range-limited lhs is NOT exact, so lhs-exact
  mode instead quantizes with per-segment scales too -- the 'grad-operand-only' recipe should
  simply cast lhs at higher precision upstream if required).
  """
  del quantize_lhs
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    # ops.py may pass a scalar (shape ()); the kernel's prefetch ref needs shape (1,)
    group_offset = jnp.asarray(group_offset, jnp.int32).reshape(1)
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
  target_zero_ref_bytes = 2 * 1024 * 1024

  lhs_q_dummy = jax.ShapeDtypeStruct(lhs.shape, jnp.float8_e4m3fn)
  rhs_q_dummy = jax.ShapeDtypeStruct(rhs.shape, jnp.float8_e4m3fn)
  cfgs = tgmm_lib.make_tgmm_configs(
      lhs_q_dummy,
      rhs_q_dummy,
      None,  # rhs_scale: None -> has_scale False; our scales ride separate operands
      group_sizes,
      num_actual_groups,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=preferred_element_type or jnp.bfloat16,
      acc_dtype=acc_dtype,
      target_zero_ref_bytes=target_zero_ref_bytes,
  )
  dims, tiles = cfgs.dims, cfgs.tiles
  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = gmm_v2.align_to(dims.size_n, num_lanes)
  aligned_k = gmm_v2.align_to(dims.size_k, tiles.tile_k)
  max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1

  # caller-side segment quantize on the SAME gm decomposition the kernel builds
  gm_offs, _ = compute_gm_metadata(
      group_sizes, tiles.tile_m, dims.size_lhs_sublane, max_num_gm
  )
  lhs_q, lhs_scales = segment_quantize(lhs, gm_offs, max_num_gm)
  rhs_q, rhs_scales = segment_quantize(rhs, gm_offs, max_num_gm)
  lhs_scales = jnp.pad(lhs_scales, ((0, 0), (0, aligned_k - dims.size_k)))[:, None, :]
  rhs_scales = jnp.pad(rhs_scales, ((0, 0), (0, aligned_n - dims.size_n)))[:, None, :]

  out_init = jax.ShapeDtypeStruct(
      (num_actual_groups, aligned_k, aligned_n), cfgs.out_dtype
  )
  scratch_shapes = [
      pltpu.VMEM((tiles.tile_k, tiles.tile_n), cfgs.acc_dtype),
      gmm_v2.MetadataRef(
          gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
          gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
      ),
  ]
  out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
  tile_zero_k = target_zero_ref_bytes // num_lanes // out_bytes
  tile_zero_k = min(tile_zero_k, dims.size_k)
  size_out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
  tile_zero_k = (tile_zero_k // size_out_sublane) * size_out_sublane
  scratch_shapes += [
      pltpu.VMEM((tile_zero_k, num_lanes), cfgs.out_dtype),
      pltpu.SemaphoreType.DMA((1,)),
  ]
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  out = pl.pallas_call(
      functools.partial(_kernel_main, cfgs=cfgs, skip_zero_out=interpret),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[hbm, hbm, hbm, hbm],
          out_specs=hbm,
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name="tgmm_block_fp8",
      cost_estimate=tgmm_lib.get_cost_estimate(cfgs),
      interpret=interpret,
  )(
      group_sizes,
      group_offset,
      lhs_q.reshape(-1, dims.size_lhs_sublane, dims.size_k),
      lhs_scales,
      rhs_q.reshape(-1, dims.size_lhs_sublane, dims.size_n),
      rhs_scales,
  )[:, : dims.size_k, : dims.size_n]
  if interpret:
    # interpret mode skipped the in-kernel zero_out: mask empty groups here
    gsz = lax.dynamic_slice_in_dim(group_sizes, group_offset[0], num_actual_groups)
    out = jnp.where((gsz > 0)[:, None, None], out, 0)
  return out
