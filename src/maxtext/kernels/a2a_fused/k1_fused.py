# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
# k1_fused.py — Kernel 1: fused a2a-in-kernel + gmm dispatch (K1_KERNEL_SPEC.md).
#
# Self-contained fork of gather/tokamax_fork/gmm_v2_smemfix.py (vendored, jax/pallas
# imports only). The vanilla gmm_v2 entry is kept intact (check_k1.py uses it, per
# segment, as the bit-exactness reference); K1 additions are marked "K1:".
#
# STRUCTURE SHIPPED: the spec's primary structure — a single grid-less pallas_call
# whose body sequences barrier -> blocked remote sends (BLK-row static-size DMAs in
# a fori_loop per dst) -> EP x (semaphore-gated per-src emit_pipeline, a STATIC
# python loop) -> send drain. The one-emit_pipeline-with-src-outer-grid fallback was
# NOT needed: both AOT shapes compile with the sequenced-pipelines body.
#
# Deviations from the spec text (documented, not redesigns):
#   * x_recv scratch memory space: spec says "memory_space ANY"; jax 0.10.1 has no
#     pltpu ANY scratch constructor — pltpu.HBM((EP*CAP, K), dtype) is the concrete
#     equivalent (unpadded HBM scratch), used instead.
#   * x_recv/y_out are handled internally as FLAT [EP*CAP, rows] buffers with the
#     per-src base offset folded into the (forked) index maps / zero-fill bounds,
#     because ref.at[src].reshape(...) (reshape of a transformed ref) is not
#     supported; gmm_a2a still returns the contract shape [EP, CAP, N].
#   * Self-send is the uniform remote-copy path (device_id=(dst,) covers dst==my at
#     runtime — my_id is traced so a trace-time special case is impossible; this is
#     exactly the proven splash_wag broadcast-to-all idiom).
#   * BLK-ALIGNED SEGMENT REPACK (replaces the spec's pad-x_send-by-BLK rule): all
#     HBM memrefs in Mosaic carry tiled layouts, so a DMA can only slice rows at
#     tile-row granularity — the ragged segment starts (prefix sums of C) are not
#     expressible, and the sub-tile row phase cannot be moved by any DMA (verified:
#     tiled-input slice rejected; HBM scratch is tiled too; HBM<->HBM copies unify
#     layouts). gmm_a2a therefore re-packs x_send ONCE at the XLA level (one row
#     gather) so dst segment d starts at sum_{d'<d} align_up(rows_d', BLK); the
#     kernel and every receiver derive the same aligned starts from C. All in-kernel
#     DMAs then run on 3-D sublane views (leading dim has no tile constraint — the
#     zero_out idiom). Receiver-side semantics are UNCHANGED (slot row 0 = first
#     row of the segment), so §3 and the bit-exactness contract hold verbatim.
#   * Spec's `pl.semaphore_wait(dma_sem, bytes)` drain: jax 0.10.1 only allows
#     REGULAR/BARRIER sems there; DMA sems are drained with the byte-equivalent
#     reconstructed-copy .wait() (traced slice size — the zero_out_end idiom).
from abc import ABC, abstractmethod
import dataclasses
import functools
from typing import Any, Callable, Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Util.


def swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
  """Activation used in some models such as GPT-OSS."""

  gate = jnp.clip(gate, max=limit)
  up = jnp.clip(up, min=-limit, max=limit)
  glu = gate * jax.nn.sigmoid(alpha * gate)
  return (up + 1.0) * glu


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
  """Applies a fused activation function to the accumulator.

  This function is used when an activation function is fused with the matrix
  multiplication. The input accumulator `acc` is expected to contain
  concatenated results for both the 'gate' and 'up' projections.

  Args:
    acc: The accumulator array, with the last dimension being 2 * tile_n.
    fuse_act: The name of the activation function to apply. Supported values are
      "silu", "gelu", and "swigluoai". If None, no activation is applied.

  Returns:
    The result of applying the activation function.

  Raises:
    NotImplementedError: If an unsupported `fuse_act` is provided.
  """

  if fuse_act is None:
    return acc

  acc_gate, acc_up = jnp.split(acc, 2, -1)
  match fuse_act:
    case "silu":
      return jax.nn.silu(acc_gate) * acc_up
    case "gelu":
      return jax.nn.gelu(acc_gate) * acc_up
    case "swigluoai":
      return swigluoai(acc_gate, acc_up)
    case _:
      raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
  return pl.cdiv(x, a) * a


# Define data classes.


class RhsRef(ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abstractmethod
  def get_weight(self) -> jax.Array:
    ...

  @abstractmethod
  def get_scale(self) -> jax.Array:
    ...

  @abstractmethod
  def get_bias(self) -> jax.Array:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
  """Dataclass for a single weights."""

  weight: Any
  scale: Any | None
  bias: Any | None

  def get_weight(self) -> jax.Array:
    return self.weight[...]

  def get_scale(self) -> jax.Array:
    assert self.scale is not None
    return self.scale[...]

  def get_bias(self) -> jax.Array:
    assert self.bias is not None
    return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
  """Dataclass for gate and up weights used in fused activation."""

  gate: WeightsRef
  up: WeightsRef

  def get_weight(self) -> jax.Array:
    w_gate = self.gate.get_weight()
    w_up = self.up.get_weight()
    return jnp.concatenate([w_gate, w_up], axis=-1)

  def get_scale(self) -> jax.Array:
    s_gate = self.gate.get_scale()
    s_up = self.up.get_scale()
    return jnp.concatenate([s_gate, s_up], axis=-1)

  def get_bias(self) -> jax.Array:
    b_gate = self.gate.get_bias()
    b_up = self.up.get_bias()
    return jnp.concatenate([b_gate, b_up], axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  # SMEM-FIX: O(num_groups) representation, M-independent (was O(num_tiles)).
  # group_start[g] = cumulative row offset of group g (cumsum of group_sizes).
  # gm_prefix[g]   = cumulative #m-tiles emitted for groups < g.
  # Both length (size_lhs_group + 1), indexed by absolute lhs_group_id.
  # (group_id, m_start, m_end) are derived per grid step in derive_tile(),
  # reproducing the old gm_id_to_* arrays. group_offset[0] adjusts lhs_group_id
  # -> the relative group_id used to index the weights.
  group_start: jax.Array
  gm_prefix: jax.Array
  group_offset: jax.Array  # int32[1]
  # bounds[0]=first processed row (compute_start), bounds[1]=last processed row
  # (compute_end). Written at STATIC indices by fill_metadata so the zero-fill
  # DMA bounds read a static-index SMEM scalar (matches the original
  # gm_id_to_m_offset[0]); a dynamic-index read here trips Mosaic tile alignment.
  bounds: jax.Array  # int32[2]


def derive_tile(metadata_ref, gm_id, cfgs):
  """Reproduce (group_id, m_start, m_end) for grid step gm_id from O(G) metadata.

  Exact closed-form equivalent of the old per-tile gm_id_to_group_id /
  gm_id_to_m_offset arrays (validated vs fill_metadata over 1080 random cases,
  incl. empty groups, sublane-misaligned starts, group_offset != 0).
  """
  sublane = cfgs.dims.size_lhs_sublane
  tile_m = cfgs.tiles.tile_m
  n_groups = metadata_ref.gm_prefix.shape[0] - 1   # = size_lhs_group (static)
  gp = metadata_ref.gm_prefix
  # lid = #{ g in [1, n_groups] : gm_prefix[g] <= gm_id }  (last group whose
  # tile-prefix does not exceed gm_id; correct across empty groups).
  def _count(g, acc):
    return acc + (gp[g + 1] <= gm_id).astype(jnp.int32)
  lid = lax.fori_loop(0, n_groups, _count, jnp.int32(0))
  lid = jnp.minimum(lid, n_groups - 1)                 # guard degenerate gm_id
  j = gm_id - gp[lid]                                  # within-group tile index
  gstart = metadata_ref.group_start[lid]
  gend = metadata_ref.group_start[lid + 1]
  base = gstart - gstart % sublane                     # sublane-aligned base
  m_start = jnp.where(j == 0, gstart, base + j * tile_m)
  m_end = jnp.minimum(base + (j + 1) * tile_m, gend)
  group_id = lid - metadata_ref.group_offset[0]        # relative -> weight index
  return group_id, m_start, m_end


@dataclasses.dataclass(frozen=True)
class TileSizes:
  tile_m: int
  tile_k: int
  tile_n: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
  size_m: int
  size_k: int
  size_n: int
  size_group: int
  size_lhs_group: int
  size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
  quant_dtype: jnp.dtype | None
  quant_block_size: int | None
  dtype: jnp.dtype
  has_bias: bool = False
  has_scale: bool = False

  @property
  def should_bitcast(self) -> bool:
    bits = jax.dtypes.itemsize_bits(self.dtype)
    return bits < 8

  @property
  def should_dequantize_before_matmul(self) -> bool:
    if not self.has_scale:
      return False
    assert self.quant_block_size is not None
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    return self.quant_block_size < mxu_size

  @property
  def should_dequantize_after_matmul(self) -> bool:
    return self.has_scale and not self.should_dequantize_before_matmul


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
  tiles: TileSizes
  dims: Dimensions
  lhs_cfgs: InputConfigs
  rhs_cfgs: InputConfigs
  out_dtype: jnp.dtype
  acc_dtype: jnp.dtype
  zero_init: bool
  fuse_act: str | None
  # PHASE E: the lhs arrives PRE-quantized (fp8 wire) with a per-row f32
  # dequant scale riding as a sidecar; the epilogue multiplies the f32
  # accumulator by it (before the out-dtype cast). False on every bf16 path
  # (default => vanilla gmm_v2 traces are unchanged).
  lhs_row_scale: bool = False

  @property
  def num_quant_blocks_per_tile_k(self) -> int:
    return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

  @property
  def out_size_n(self) -> int:
    if self.fuse_act is None:
      return self.dims.size_n
    else:
      return self.dims.size_n // 2


TileFn = Callable[
    [Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes
]


class IndexMaps:
  """Index maps for GMM kernel.

  K1: `row_base` (rows in size_lhs_sublane units, may be traced) offsets the
  lhs/out row indices so one flat [EP*CAP, ...] buffer can host EP per-src
  segments; the metadata (and derive_tile) stay segment-relative. Vanilla
  gmm_v2 passes 0 (identical behavior).

  K1 self-segment direct read: `lhs_row_base` (None -> row_base) lets the lhs
  live in a DIFFERENT buffer than the output — the src==my pipeline reads its
  lhs straight from the BLK-aligned x_send buffer at my aligned segment start
  while the output still lands in y_out slot my (row_base).
  """

  def __init__(
      self,
      metadata_ref: MetadataRef,
      cfgs: GmmConfigs,
      row_base: Any = 0,
      lhs_row_base: Any = None,
  ):
    self.metadata_ref = metadata_ref
    self.cfgs = cfgs
    self.row_base = row_base
    self.lhs_row_base = row_base if lhs_row_base is None else lhs_row_base

  def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    _, m_start, m_end = derive_tile(self.metadata_ref, gm_id, self.cfgs)

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start

    return (pl.ds(self.lhs_row_base + row_start, row_size), 0, k_id)

  def lhs_scale_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
    """PHASE E: the per-row scale sidecar tile for the SAME rows as the lhs.

    The sidecar buffer is [rows, num_lanes] f32 (each row's scale replicated
    across the full lane row — see gmm_a2a's fp8 notes), 3-D-viewed with the
    SAME size_lhs_sublane grouping as the lhs, so the row indexing (and
    lhs_row_base) is unit-for-unit identical to lhs_index_map. Scales are
    k-invariant (k_id dropped) and one 128-lane block wide (col index 0).
    """
    del k_id
    _, m_start, m_end = derive_tile(self.metadata_ref, gm_id, self.cfgs)
    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_size = row_end - row_start
    return (pl.ds(self.lhs_row_base + row_start, row_size), 0, 0)

  def rhs_weight_index_map(
      self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array
  ):
    group_id, _, _ = derive_tile(self.metadata_ref, gm_id, self.cfgs)
    return (group_id, k_id, n_id)

  def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    group_id, _, _ = derive_tile(self.metadata_ref, gm_id, self.cfgs)
    return (group_id, 0, n_id)

  def rhs_scale_index_map(
      self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array
  ):
    group_id, _, _ = derive_tile(self.metadata_ref, gm_id, self.cfgs)
    # Simply multiplying k_id by num_quant_blocks_per_tile_k will not work
    # since a single quant block could be shared along multiple k tile.
    k_row = k_id * self.cfgs.tiles.tile_k
    b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
    b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
    return (group_id, b_tile_id, 0, n_id)

  def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
    is_last_gm = gm_id == (pl.num_programs(1) - 1)
    _, m_start, m_end = derive_tile(self.metadata_ref, gm_id, self.cfgs)

    row_start = m_start // self.cfgs.dims.size_lhs_sublane
    capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
    last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
    row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
    row_size = row_end - row_start

    return (pl.ds(self.row_base + row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
    row_base: Any = 0,
    lhs_row_base: Any = None,
) -> Tuple[Tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
  """Generates block specs for the given lhs, rhs, and out refs."""

  index_map = IndexMaps(metadata_ref, cfgs, row_base, lhs_row_base)
  bounded_slice_gm = pl.BoundedSlice(
      cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane
  )

  lhs_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
      index_map.lhs_index_map,
  )

  tile_k_rhs = cfgs.tiles.tile_k
  if cfgs.rhs_cfgs.should_bitcast:
    packing = pl.cdiv(32, jax.dtypes.itemsize_bits(cfgs.rhs_cfgs.dtype))
    tile_k_rhs //= packing

  rhs_weight_spec = pl.BlockSpec(
      (None, tile_k_rhs, cfgs.tiles.tile_n),
      index_map.rhs_weight_index_map,
      pipeline_mode=pl.Buffered(buffer_count=3),
  )
  rhs_scale_block_spec = rhs_bias_block_spec = None
  if cfgs.rhs_cfgs.has_bias:
    rhs_bias_block_spec = pl.BlockSpec(
        (None, 1, cfgs.tiles.tile_n),
        index_map.rhs_bias_index_map,
    )
  if cfgs.rhs_cfgs.has_scale:
    rhs_scale_block_spec = pl.BlockSpec(
        (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
        index_map.rhs_scale_index_map,
    )

  rhs_block_spec = WeightsRef(
      weight=rhs_weight_spec,
      scale=rhs_scale_block_spec,
      bias=rhs_bias_block_spec,
  )

  out_block_spec = pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
      index_map.out_index_map,
  )

  return (lhs_block_spec, rhs_block_spec), out_block_spec


def make_lhs_scale_spec(
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
    scale_row_base: Any,
) -> pl.BlockSpec:
  """PHASE E: block spec for the per-row scale sidecar pipeline input.

  Mirrors the lhs block spec (same bounded m-slice, same sublane grouping,
  same row base semantics) with a fixed one-lane-tile width. scale_row_base is
  the sidecar-buffer row offset of this segment in size_lhs_sublane units —
  identical to the lhs row base because both buffers share the row layout.
  """
  index_map = IndexMaps(metadata_ref, cfgs, scale_row_base)
  bounded_slice_gm = pl.BoundedSlice(
      cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane
  )
  num_lanes = pltpu.get_tpu_info().num_lanes
  return pl.BlockSpec(
      (bounded_slice_gm, cfgs.dims.size_lhs_sublane, num_lanes),
      index_map.lhs_scale_index_map,
  )


# Define kernels.


def inner_kernel(
    # In
    tiled_lhs_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_k]
    tiled_rhs_ref: RhsRef,  # [tile_k, tile_n]
    # Out
    tiled_out_ref: jax.Array,
    # [tile_m // size_lhs_sublane, size_lhs_sublane, tile_n]
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
    tiled_lhs_scale_ref: jax.Array | None = None,
    # PHASE E: [tile_m // size_lhs_sublane, size_lhs_sublane, num_lanes] f32 —
    # the per-row dequant scale sidecar tile (each row's scale replicated
    # across all lanes). Only set when cfgs.lhs_row_scale.
):
  """Inner kernel invoked by emit_pipeline to perform matmul.

  tiled_lhs_ref and tiled_out_ref points to rows [m_start:m_end] of lhs and out.
  Additionally, m_start and m_end does not have to align with tile boundaries
  [m_offset:m_offset+tile_m]. Therefore, rows [m_offset:m_start] and
  [m_end:m_offset+tile_m] of tiled_lhs_ref and tiled_out_ref will contain
  invalid data and needs to be masked out.

  Args:
    tiled_lhs_ref: Contains value lhs[m_start:m_end, k_start:k_end]
    tiled_rhs_ref: Contains value rhs[g_id, k_start:k_end, n_start:n_end]. where
      g_id is the group associated with lhs[m_start:m_end, :]
    tiled_out_ref: Contains value out[m_start:m_end, n_start:n_end]
    partial_out_ref: Contains last size_lhs_sublane rows of the previous output.
      Will be initialized to zero if this is first tile for grid[n_id, :, :].
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    cfgs: GmmConfigs.
  """

  def _matmul(is_first_k_step: bool, is_last_k_step: bool):
    tpu_info = pltpu.get_tpu_info()
    mxu_size = tpu_info.mxu_column_size

    # PHASE E FAST PATH (static, cfgs-only -> its own trace): the lhs arrives
    # ALREADY fp8-quantized (dtype float8_e4m3fn) carrying a per-ROW dequant
    # scale sidecar (cfgs.lhs_row_scale). A per-row scale is K-UNIFORM, so it
    # factors OUT of the K contraction; the blocked unquantized path below
    # instead re-runs a per-rhs-quant-block matmul + VPU pop/scale/add per K
    # sub-block (14 blocks at K=7168/qblk=512 -> the measured VALU storm on
    # already-quantized data). Requiring a FULL-CHANNEL rhs scale (num_blocks
    # == 1: quant_block_size == size_k) makes the rhs scale K-uniform too, so
    # BOTH scales can be applied ONCE in the epilogue after a SINGLE matmul per
    # mxu-n-block over the full tile_k (MXU accumulates internally). Gated on
    # (lhs_row_scale AND lhs is fp8 AND rhs dequant-after-matmul) only; every
    # bf16 / in-kernel-quant trace is byte-identical (this branch not taken).
    fast_fp8 = (
        cfgs.lhs_row_scale
        and cfgs.lhs_cfgs.quant_dtype is None
        and jnp.issubdtype(cfgs.lhs_cfgs.dtype, jnp.floating)
        and jax.dtypes.itemsize_bits(cfgs.lhs_cfgs.dtype) == 8
        and cfgs.rhs_cfgs.should_dequantize_after_matmul
    )
    if fast_fp8:
      assert cfgs.rhs_cfgs.quant_block_size == cfgs.dims.size_k, (
          "fp8 row-scale fast path REQUIRES a full-channel rhs scale "
          "(num_blocks == 1); got quant_block_size="
          f"{cfgs.rhs_cfgs.quant_block_size} != size_k={cfgs.dims.size_k}. "
          "Sub-channel rhs scales are incompatible: their per-K-block factors "
          "do not commute out of the contraction."
      )

    # Step 1: Input pre-processing.
    tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[...]
    tiled_rhs = tiled_rhs_ref.get_weight()
    # When rhs is packed (quantized dtype packed into uint32), unpack it
    # back to the original dtype using pltpu.bitcast which operates on K
    # axis. This expands the K dimension back to tile_k.
    if cfgs.rhs_cfgs.should_bitcast:
      tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.dtype)
    rhs_tile_n = tiled_rhs.shape[1]

    # This should only be taken in the case where we don't requantize
    # the scales and thus we need to dequantize inside VMEM to avoid small
    # contracting dimmensions
    if cfgs.rhs_cfgs.should_dequantize_before_matmul:
      rhs_qbs = cfgs.rhs_cfgs.quant_block_size
      tiled_rhs_scale = tiled_rhs_ref.get_scale().astype(acc_ref.dtype)
      num_blocks = cfgs.num_quant_blocks_per_tile_k
      tiled_rhs_dequant = tiled_rhs.astype(acc_ref.dtype).reshape(
          num_blocks, rhs_qbs, rhs_tile_n
      )
      tiled_rhs_dequant = tiled_rhs_dequant * tiled_rhs_scale
      tiled_rhs = tiled_rhs_dequant.reshape(cfgs.tiles.tile_k, rhs_tile_n)

    valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
    if is_last_k_step and valid_k != 0:
      mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
      tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

    # Step 2: Matmul.
    acc_list = []
    if cfgs.lhs_cfgs.quant_dtype is None:
      # Unquantized matmul path.
      rhs_qbs = cfgs.rhs_cfgs.quant_block_size

      # PHASE E: fp8 operands on hardware without fp8 MXU support (the local
      # v4 correctness box rejects f8E4M3FN matmuls with Mosaic E2001) — cast
      # to bf16 for the matmul. EXACT: e4m3 carries a 4-bit significand, so
      # every e4m3 value (and product, <= 8 significand bits) is exactly
      # representable in bf16; accumulation stays preferred_element_type=f32
      # either way. Static trace-time no-op on v7x and for all bf16 inputs
      # (bf16 paths lower byte-identically). Mirrors the quantized path's
      # existing is_matmul_supported cast.
      if not tpu_info.is_matmul_supported(tiled_lhs.dtype, tiled_rhs.dtype):
        if jnp.issubdtype(tiled_lhs.dtype, jnp.floating) and (
            jax.dtypes.itemsize_bits(tiled_lhs.dtype) == 8
        ):
          tiled_lhs = tiled_lhs.astype(jnp.bfloat16)
        if jnp.issubdtype(tiled_rhs.dtype, jnp.floating) and (
            jax.dtypes.itemsize_bits(tiled_rhs.dtype) == 8
        ):
          tiled_rhs = tiled_rhs.astype(jnp.bfloat16)

      if fast_fp8:
        # ONE matmul per mxu-n-block over the FULL tile_k: no K sub-block loop,
        # no per-block VPU pop/scale/add. The full-channel rhs scale and the
        # per-row scale are both applied once in the is_last_k_step epilogue.
        for start_n in range(0, rhs_tile_n, mxu_size):
          end_n = min(rhs_tile_n, start_n + mxu_size)
          acc_n = jnp.matmul(
              tiled_lhs,
              tiled_rhs[:, start_n:end_n],
              preferred_element_type=jnp.float32,
          ).astype(acc_ref.dtype)
          acc_list.append(acc_n)
      else:
        for start_n in range(0, rhs_tile_n, mxu_size):
          end_n = min(rhs_tile_n, start_n + mxu_size)
          col_size = end_n - start_n

          acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
          for b_id in range(cfgs.num_quant_blocks_per_tile_k):
            start_k = b_id * rhs_qbs
            end_k = start_k + rhs_qbs

            block_acc = jnp.matmul(
                tiled_lhs[:, start_k:end_k],
                tiled_rhs[start_k:end_k, start_n:end_n],
                preferred_element_type=jnp.float32,
            ).astype(acc_ref.dtype)

            if cfgs.rhs_cfgs.should_dequantize_after_matmul:
              tiled_rhs_scale = tiled_rhs_ref.get_scale()
              block_acc *= tiled_rhs_scale[b_id, :, start_n:end_n].astype(
                  acc_ref.dtype
              )

            acc_n += block_acc
          acc_list.append(acc_n)
    else:
      # Quantized matmul path.
      lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
      q_block_size = cfgs.lhs_cfgs.quant_block_size

      if jnp.issubdtype(lhs_q_dtype, jnp.floating):
        dtype_max = float(jnp.finfo(lhs_q_dtype).max)
        preferred_element_type = jnp.float32
      else:
        dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
        preferred_element_type = jnp.int32

      # Without n outer loop, result of quantized matmul becomes available only
      # at the last iteration of the loop. This means [tile_m, tile_n] value
      # needs to be stored until the last iteration. By adding n outer loop,
      # result of [tile_m, mxu_size] becomes available at the end of every k
      # inner loop which can be used to pipeline subsequent VPU or VST ops with
      # MXU ops for the next [tile_m, mxu_size].
      for start_n in range(0, rhs_tile_n, mxu_size):
        end_n = min(rhs_tile_n, start_n + mxu_size)
        col_size = end_n - start_n

        acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
        for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
          end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

          block_lhs = tiled_lhs[:, start_k:end_k]
          block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

          # Perform lhs quantization. Note that for every block_lhs,
          # same computation will be performed tiles_n//mxu_size times.
          # But we can let compiler perform CSE and avoid recomputation.
          block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
          block_scale = block_abs_max / dtype_max

          # If block_scale=0, it will cause division by zero and return either
          # NaN or Inf. Since this can cause numeric issue when downcasting to
          # quantized value, we convert them into 0.
          block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
          # Convert lhs into quantized dtype.
          block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

          # Unlike unquantized path, compiler may not perform implicit type
          # conversion due to numeric concerns. As this can cause unsupported
          # matmul error, explicit type conversion is performed.
          if not tpu_info.is_matmul_supported(lhs_q_dtype, block_rhs.dtype):
            block_rhs = block_rhs.astype(lhs_q_dtype)

          block_acc = jnp.matmul(
              block_lhs_q,
              block_rhs,
              preferred_element_type=preferred_element_type,
          ).astype(acc_ref.dtype)

          block_acc *= block_scale.astype(acc_ref.dtype)

          # Apply rhs subchannel scale per quant block.
          if cfgs.rhs_cfgs.should_dequantize_after_matmul:
            b_id = start_k // cfgs.rhs_cfgs.quant_block_size
            rhs_scale_slice = tiled_rhs_ref.get_scale()
            block_acc *= rhs_scale_slice[b_id, :, start_n:end_n].astype(
                acc_ref.dtype
            )

          acc_n += block_acc
        acc_list.append(acc_n)
    acc = jnp.concatenate(acc_list, axis=1)

    # Step 3: Output post-processing.
    if not is_first_k_step:
      acc += acc_ref[...]

    if is_last_k_step:
      # PHASE E: apply the per-row lhs dequant scale to the fully-accumulated
      # f32 acc (before bias/act/cast). The sidecar tile holds each row's
      # scale replicated across every lane, so the lane-reduce (keepdims)
      # yields the exact scale as a [tile_m, 1] column — the SAME broadcast
      # idiom the in-kernel lhs-quant path uses for block_scale (a keepdims
      # reduce, never a [:, None]/trailing-1 reshape — the header law).
      # Rows outside [m_start, m_end) may carry garbage scales; the group
      # mask below jnp.where-selects them to 0 (kills NaN too).
      if cfgs.lhs_row_scale:
        assert tiled_lhs_scale_ref is not None
        scale_rows = tiled_lhs_scale_ref.reshape(
            -1, tiled_lhs_scale_ref.shape[-1]
        )[...]  # [tile_m, num_lanes], every lane = the row's scale
        scale_col = jnp.max(scale_rows, axis=1, keepdims=True)  # [tile_m, 1]
        acc = acc * scale_col.astype(acc.dtype)
        if fast_fp8:
          # Fold the full-channel rhs dequant scale into the SAME epilogue pass
          # (the fast path deferred it here instead of scaling per K sub-block).
          # num_blocks == 1 -> get_scale() is [1, 1, out_size_n]; [0] drops the
          # leading block dim -> [1, out_size_n] (the exact index idiom the
          # blocked path uses for tiled_rhs_scale[b_id], NOT a [:, None] /
          # trailing-1 reshape). Broadcasts across rows against [tile_m, N].
          rhs_scale_row = tiled_rhs_ref.get_scale()[0].astype(acc.dtype)
          acc = acc * rhs_scale_row

      if cfgs.rhs_cfgs.has_bias:
        tiled_rhs_bias = tiled_rhs_ref.get_bias()
        acc += tiled_rhs_bias.astype(acc.dtype)

      acc = apply_act_fn(acc, cfgs.fuse_act)

      gm_id = pl.program_id(1)

      # Mask out rows that does not belong to the current group.
      _, m_start, m_end = derive_tile(metadata_ref, gm_id, cfgs)
      m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

      m_start_local = m_start - m_offset
      m_end_local = m_end - m_offset

      iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
      mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
      acc_masked = jnp.where(mask, acc, 0).reshape(tiled_out_ref.shape)

      # Write the final output to the output ref.
      tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

      # If this is the first tile for grid[n_id, :, :], we initialize the
      # partial out to zeros. Otherwise, partial out from last tile of
      # grid[n_id-1, :, :] can be used and cause numeric issues.
      partial_out_zeros = jnp.zeros_like(partial_out_ref)

      # Accumulate the partial output from the previous step.
      tiled_out_ref[0] += jnp.where(
          gm_id == 0, partial_out_zeros, partial_out_ref[...]
      )

      # Consider following case where size_lhs_sublane = 4, number denotes group
      # id and | denotes boundaries between sublanes:
      # | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 |
      #
      # Assuming group id of current step is 1, current step will not completely
      # fill size_lhs_sublane rows and will be revisited at the next step. By
      # storing the partial rows into the partial_out_ref, the next step can
      # read them and accumulate to them.  Additionally, for group id of 2,
      # since it completely fills the size_lhs_sublane rows, we need to zero out
      # partial_out_ref to avoid numeric error for group 3.
      last_row = m_end_local // cfgs.dims.size_lhs_sublane
      partial_out_ref[...] = jnp.where(
          m_end_local % cfgs.dims.size_lhs_sublane == 0,
          partial_out_zeros,
          tiled_out_ref[last_row],
      )
    else:
      acc_ref[...] = acc

  # Define matmul wrapper functions.
  @jax.named_scope("matmul_first_last")
  def matmul_first_last():
    _matmul(is_first_k_step=True, is_last_k_step=True)

  @jax.named_scope("matmul_first")
  def matmul_first():
    _matmul(is_first_k_step=True, is_last_k_step=False)

  @jax.named_scope("matmul")
  def matmul():
    _matmul(is_first_k_step=False, is_last_k_step=False)

  @jax.named_scope("matmul_last")
  def matmul_last():
    _matmul(is_first_k_step=False, is_last_k_step=True)

  # Select and execute matmul function based on the current step.
  num_k = pl.num_programs(2)
  k_id = pl.program_id(2)

  is_first_k_step = k_id == 0
  is_last_k_step = k_id == (num_k - 1)

  lax.cond(
      is_first_k_step,
      lambda: lax.cond(
          is_last_k_step,
          matmul_first_last,
          matmul_first,
      ),
      lambda: lax.cond(
          is_last_k_step,
          matmul_last,
          matmul,
      ),
  )


def inner_kernel_rs(
    tiled_lhs_ref: jax.Array,
    tiled_rhs_ref: RhsRef,
    tiled_lhs_scale_ref: jax.Array,
    tiled_out_ref: jax.Array,
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
  """PHASE E: inner_kernel with the per-row scale sidecar as a THIRD pipeline
  input (emit_pipeline passes inputs positionally in in_specs order)."""
  inner_kernel(
      tiled_lhs_ref,
      tiled_rhs_ref,
      tiled_out_ref,
      partial_out_ref,
      acc_ref,
      metadata_ref,
      cfgs=cfgs,
      tiled_lhs_scale_ref=tiled_lhs_scale_ref,
  )


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group] (K1: int32[>=base+G])
    group_offset_ref: jax.Array | None,  # int32[1] (K1: None -> 0)
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
    base: Any = 0,  # K1: scalar (traced) offset into the 1-D prefetch ref; the
    # group sizes of this segment are lhs_group_sizes_ref[base + lid],
    # lid in [0, size_lhs_group). Arithmetic 1-D SMEM indexing only.
) -> jax.Array:
  """Fills the metadata for the given lhs group sizes and group offset.

  Iterates over the lhs group sizes and if the group id is valid, determines
  the number of gm tiles that are needed to process the current group. Then,
  it fills starting and ending offset (gm_id_to_m_offset), and the group id
  (gm_id_to_group_id) for each gm tile.

  Args:
    lhs_group_sizes_ref: The group sizes of lhs.
    group_offset_ref: Offset of the first group to process.
    metadata_ref: Metadata that is used to determine the group id and m offsets
      for each gmm tile.
    cfgs: GmmConfigs.

  Returns:
      The number of gm tiles to process lhs with given group offset.
  """

  # SMEM-FIX: fill O(num_groups) arrays in one pass instead of O(num_tiles).
  #   group_start[lid+1] = cumsum of group_sizes        (absolute row offset)
  #   gm_prefix[lid+1]   = cumsum of tiles-per-group     (#m-tiles emitted)
  # tiles-per-group uses the same sublane-aligned cdiv(group_size+local_offset,
  # tile_m) as the original inner loop; derive_tile() reconstructs the exact
  # per-tile (group_id, m_start, m_end) from these (validated vs the original
  # over 1080 random cases, incl. empty groups, misalignment, group_offset!=0).
  if group_offset_ref is None:  # K1: per-src segments always start at group 0
    group_offset = jnp.int32(0)
  else:
    group_offset = group_offset_ref[0]
  # K1: the prefetch ref may hold EP*EP*E_local counts; the loop bound is the
  # per-call group count from cfgs (== ref.shape[0] on the vanilla gmm_v2 path).
  size_lhs_group = cfgs.dims.size_lhs_group              # static
  size_group = cfgs.dims.size_group
  tile_m = cfgs.tiles.tile_m
  sublane = cfgs.dims.size_lhs_sublane

  metadata_ref.group_offset[0] = group_offset
  metadata_ref.group_start[0] = jnp.int32(0)
  metadata_ref.gm_prefix[0] = jnp.int32(0)

  def body(lid, carry):
    start_m_offset, num_gm = carry
    group_id = lid - group_offset
    group_size = lhs_group_sizes_ref[base + lid]
    local_offset = start_m_offset % sublane
    curr_num_gm = pl.cdiv(group_size + local_offset, tile_m)
    # process only groups in [group_offset, group_offset + size_group) (the loop
    # bound in the original) with non-empty size.
    should_process = jnp.logical_and(
        group_size > 0,
        jnp.logical_and(group_id >= 0, group_id < size_group),
    )
    curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
    next_m_offset = start_m_offset + group_size
    next_num_gm = num_gm + curr_num_gm
    metadata_ref.group_start[lid + 1] = next_m_offset
    metadata_ref.gm_prefix[lid + 1] = next_num_gm
    return next_m_offset, next_num_gm

  _, num_gm = lax.fori_loop(
      0, size_lhs_group, body, (jnp.int32(0), jnp.int32(0))
  )
  # Stash the processed-row range at static indices for the zero-fill DMA.
  metadata_ref.bounds[0] = metadata_ref.group_start[group_offset]
  metadata_ref.bounds[1] = metadata_ref.group_start[group_offset + size_group]
  return num_gm


def zero_out_start(
    out_ref: jax.Array,  # [size_m, size_n] (K1: flat [EP*CAP, size_n])
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array,  # [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    cfgs: GmmConfigs,
    row_offset: Any = 0,  # K1: first row of this src's segment in out_ref
    # (traced; must be a multiple of the out sublane tiling). Metadata bounds
    # are segment-relative and get shifted by this.
    num_rows: int | None = None,  # K1: segment length in rows (= CAP). None ->
    # whole buffer (vanilla gmm_v2 path).
):
  """Zero out output rows that are not used in the computation."""
  dims = cfgs.dims

  num_lanes = pltpu.get_tpu_info().num_lanes
  assert num_lanes == zero_ref.shape[-1]
  zero_ref[...] = jnp.zeros_like(zero_ref)

  # The zero/out buffers are out_dtype, so they must be reshaped using the
  # OUTPUT's sublane tiling -- not the lhs's. They differ when lhs_dtype !=
  # out_dtype (e.g. the dlhs backward: f32 lhs -> sublane 8, bf16 out ->
  # sublane 16); using size_lhs_sublane (8) on a bf16 VMEM buffer (tile 16)
  # trips "Expected the 2nd minor dimension is aligned to the tile".
  out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)

  zero_dma = zero_ref.reshape(-1, out_sublane, num_lanes)
  out_dma = out_ref.reshape(-1, out_sublane, out_ref.shape[-1])
  row_size = zero_dma.shape[0]

  # Row range covered by the processed groups [group_offset, +size_group).
  # These are direct SMEM reads (group_start is the cumsum of group_sizes), not
  # the derive_tile search -> the zero-fill DMA bounds stay provably tile-aligned
  # for Mosaic. Equivalent to the old gm_id_to_m_offset[0] / [num_gm] because
  # trailing/leading empty groups are zero-width in group_start.
  compute_start = metadata_ref.bounds[0]   # static-index reads (see MetadataRef)
  compute_end = metadata_ref.bounds[1]

  # K1: shift the (segment-relative) bounds into the flat buffer and clamp the
  # right fill to the end of THIS segment, not the whole buffer.
  base_dma = row_offset // out_sublane
  if num_rows is None:
    seg_end_dma = out_dma.shape[0]
  else:
    seg_end_dma = base_dma + num_rows // out_sublane

  left_zero_start = base_dma
  left_zero_end = base_dma + compute_start // out_sublane
  left_zero_size = left_zero_end - left_zero_start
  left_num_loops = pl.cdiv(left_zero_size, row_size)

  right_zero_start = base_dma + pl.cdiv(compute_end, out_sublane)
  right_zero_end = seg_end_dma
  right_zero_size = right_zero_end - right_zero_start
  right_num_loops = pl.cdiv(right_zero_size, row_size)

  def fill_zero(i, zero_size, *, start, end):
    dma_start = start + i * row_size
    dma_end = jnp.minimum(dma_start + row_size, end)
    dma_size = dma_end - dma_start

    # Static loop. Will be unrolled during compile time.
    for n_start in range(0, out_dma.shape[-1], num_lanes):
      n_end = n_start + num_lanes
      pltpu.make_async_copy(
          src_ref=zero_dma.at[pl.ds(0, dma_size)],
          dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
          sem=semaphore_ref.at[0],
      ).start(priority=1)

    return zero_size + dma_size

  @jax.named_scope("left_fill_zero")
  def left_fill_zero(i, zero_size):
    return fill_zero(i, zero_size, start=left_zero_start, end=left_zero_end)

  @jax.named_scope("right_fill_zero")
  def right_fill_zero(i, zero_size):
    return fill_zero(i, zero_size, start=right_zero_start, end=right_zero_end)

  zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
  zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
  return zero_size


def zero_out_end(
    out_ref: jax.Array,  # [size_m, size_n]
    semaphore_ref: jax.Array,  # [1]
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
  # SMEM-FIX: reshape the out_dtype buffer with its OWN sublane (see zero_out_start).
  out_sublane = pltpu.get_tpu_info().get_sublane_tiling(out_ref.dtype)
  out_dma = out_ref.reshape(-1, out_sublane, out_ref.shape[-1])
  pltpu.make_async_copy(
      src_ref=out_dma.at[pl.ds(0, zero_size)],
      dst_ref=out_dma.at[pl.ds(0, zero_size)],
      sem=semaphore_ref.at[0],
  ).wait()


def kernel_main(
    # Scalar prefetch
    lhs_group_sizes_ref: jax.Array,  # int32[size_lhs_group]
    group_offset_ref: jax.Array,  # int32[1]
    # In
    lhs_ref: jax.Array,  # [size_m, size_k]
    rhs_ref: WeightsRef,  # [size_group, size_k, size_n]
    # Out
    out_ref: jax.Array,  # [size_m, size_n]
    # Scratch memory
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n]
    acc_ref: jax.Array,  # [tile_m, tile_n]
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,  # [tile_zero_m, num_lanes]
    semaphore_ref: jax.Array | None,  # [1]
    *,
    cfgs: GmmConfigs,
):
  """Entry point for GMM kernel.

  Computes metadata to determine which rows of lhs needs processing and how
  they will be tiled. And then, invoke inner kernel using metadata.

  Uses the following notation:
  - g: rhs group dimension
  - m: Batch dimension
  - gm: Batch tiling dimension. Aligned to size_lhs_sublane and has tile size
    of tile_m. Skips over empty groups and accounts for revisited tiles.
  - k: in dimension
  - n: out dimension

  Args:
    lhs_group_sizes_ref: Reference to the group sizes of lhs.
    group_offset_ref: Reference to the group offset.
    lhs_ref: Reference to the lhs.
    rhs_ref: Reference to the rhs.
    out_ref: Reference to the out.
    partial_out_ref: Reference to the partial output.
    acc_ref: Reference to the accumulator.
    metadata_ref: Reference to the metadata.
    zero_ref: Scratch memory for storing zero values used in initialization.
    semaphore_ref: Semaphore for zero initialization DMAs.
    cfgs: GmmConfigs.
  """

  num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

  # Pack along K (2nd minor dim) so that pltpu.bitcast can unpack inside the
  # kernel.
  # [G, K, N] -> [G, K//packing, N] uint32
  if cfgs.rhs_cfgs.should_bitcast:
    rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
    rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

  # Fill metadata buffer and return number of group & m interations.
  num_gm = fill_metadata(
      lhs_group_sizes_ref,
      group_offset_ref,
      metadata_ref,
      cfgs=cfgs,
  )

  if cfgs.zero_init:
    zero_size = zero_out_start(
        out_ref,
        zero_ref,
        semaphore_ref,
        metadata_ref,
        num_gm,
        cfgs=cfgs,
    )

  (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

  if cfgs.fuse_act is not None:
    rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
    rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)

    rhs_spec = FusedWeightsRef(
        gate=rhs_spec,
        up=rhs_spec,
    )

  # Execute the inner kernel.
  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(inner_kernel, cfgs=cfgs),
      grid=(num_n, num_gm, num_k),
      in_specs=(lhs_spec, rhs_spec),
      out_specs=out_spec,
  )

  # Bounded slice requires second last dim to be aligned to the sublane size.
  # rhs_ref uses static tiling thus reshape is not needed.
  lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
  out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
  scratches = [partial_out_ref, acc_ref, metadata_ref]
  pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

  if cfgs.zero_init:
    zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
  """Calculate optimal tile sizes for GMM kernel."""

  lhs_dtype = lhs_cfgs.quant_dtype or lhs_cfgs.dtype
  rhs_dtype = rhs_cfgs.dtype

  lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

  # When using bf16 for lhs and rhs, 128 is the largest tile_m value that is
  # safe to use for most scenarios. But if lower bitwidth is used, we need
  # to tweak tile_m to account for using faster hardware unit.
  # TODO: Account for different TPU hardware specs.
  bf16_bf16_tile_m = 128
  lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
  rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
  tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod
  tile_m = min(tile_m, dims.size_m)

  # To avoid stalling MXU, we add some buffer room where tile_n cannot go
  # smaller than 2x of mxu_column_size.
  tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
  tile_n_limit = min(tile_n_limit, dims.size_n)

  size_n_per_rhs = dims.size_n
  fuse_act_factor = 1
  if fuse_act is not None:
    # When computing activation function, rhs is concatenated along dim n.
    fuse_act_factor = 2
    size_n_per_rhs //= fuse_act_factor
    tile_n_limit //= fuse_act_factor

  def _is_tile_k_quant_block_compatible(tk: int) -> bool:
    if (
        tk % rhs_cfgs.quant_block_size != 0
        and rhs_cfgs.quant_block_size % tk != 0
    ):
      return False
    return True

  # Initialize tile_k and tile_n to their maximum valid values.
  num_k_tiles = num_n_tiles = 1
  num_lanes = pltpu.get_tpu_info().num_lanes
  tile_k = align_to(dims.size_k, num_lanes)
  tile_n = align_to(size_n_per_rhs, num_lanes)

  def _gmm_vmem_estimate(tn: int, tk: int) -> int:
    # 1. LHS tile (double-buffered)
    lhs_tile_bytes = lhs_bits // 8
    lhs_vmem = 2 * tile_m * tk * lhs_tile_bytes

    # 2. RHS tile (triple-buffered, includes scale and bias if present)
    # If fuse_act is enabled, we have both gate and up weights,
    # so RHS memory is doubled.
    rhs_weight_vmem = tk * tn * rhs_bits // 8
    rhs_scale_vmem = 0
    if rhs_cfgs.has_scale and rhs_cfgs.quant_block_size is not None:
      num_quant_blocks_per_tile_k = pl.cdiv(tk, rhs_cfgs.quant_block_size)
      rhs_scale_vmem = num_quant_blocks_per_tile_k * tn * 4
    rhs_bias_vmem = 0
    if rhs_cfgs.has_bias:
      rhs_bias_vmem = tn * 4
    rhs_vmem = fuse_act_factor * (
        3 * rhs_weight_vmem + 2 * rhs_scale_vmem + 2 * rhs_bias_vmem
    )

    # 3. Accumulator
    acc_cols = fuse_act_factor * tn
    acc_dtype_bytes = 2 if lhs_cfgs.quant_dtype is not None else 4
    acc_vmem = tile_m * acc_cols * acc_dtype_bytes

    # 4. Output tile (double-buffered)
    out_dtype_bytes = jax.dtypes.itemsize_bits(lhs_cfgs.dtype) // 8
    out_vmem = 2 * tile_m * tn * out_dtype_bytes

    return lhs_vmem + rhs_vmem + acc_vmem + out_vmem

  # Multiple k tiles will introduce accumulation overhead. Thus, we first try
  # to fit the tensors into vmem by only adjusting tile_n.

  # Decrease tile_n until total memory fits in vmem limit.
  while (
      _gmm_vmem_estimate(tile_n, tile_k) > vmem_limit_bytes
      and tile_n > tile_n_limit
  ):
    num_n_tiles += 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

  # If decreasing tile_n is no longer possible, we decrease tile_k instead.
  if tile_n < tile_n_limit:
    num_n_tiles -= 1
    tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    # Decrease tile_k until total memory fits in vmem limit and tile_k is valid.
    while _gmm_vmem_estimate(
        tile_n, tile_k
    ) > vmem_limit_bytes or not _is_tile_k_quant_block_compatible(tile_k):
      num_k_tiles += 1
      tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

  if tile_n == 0 or tile_k == 0:
    final_estimate = _gmm_vmem_estimate(tile_n, tile_k)
    raise ValueError(
        f"Could not find valid tile sizes for {dims=} and"
        f" {final_estimate=} (limit: {vmem_limit_bytes})."
    )

  return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
) -> Dimensions:
  """Validates the inputs for the GMM kernel."""

  size_m = lhs.shape[0]
  size_group, size_k, size_n = rhs.shape
  size_lhs_group = group_sizes.shape[0]

  assert size_group <= size_lhs_group
  assert lhs.shape == (size_m, size_k)
  assert rhs.shape == (size_group, size_k, size_n)
  if rhs_bias is not None:
    assert rhs_bias.shape == (size_group, 1, size_n)
  if rhs_scale is not None:
    num_quant_blocks = rhs_scale.shape[1]
    assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
    assert size_k % num_quant_blocks == 0

  assert group_offset.shape == (1,)

  size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
  size_lhs_sublane = min(size_lhs_sublane, size_m)
  if fuse_act is not None:
    num_lanes = pltpu.get_tpu_info().num_lanes
    if size_n % (2 * num_lanes) != 0:
      raise ValueError(
          f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is "
          "enabled since we need to split n dimension for gate and up."
      )

  return Dimensions(
      size_m=size_m,
      size_k=size_k,
      size_n=size_n,
      size_group=size_group,
      size_lhs_group=size_lhs_group,
      size_lhs_sublane=size_lhs_sublane,
  )


def get_cost_estimate(cfgs: GmmConfigs):
  """Returns the cost estimate for the GMM kernel."""

  dims = cfgs.dims
  lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
  rhs_dtype = cfgs.rhs_cfgs.dtype

  # We use bits for rhs since it could sub-byte dtype like int4.
  rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
  fp32_bytes = jnp.dtype(jnp.float32).itemsize

  # TODO: Add compute flops for quant, dequant, and bias.
  flops = 2 * dims.size_m * dims.size_k * dims.size_n

  lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

  rhs_size = dims.size_group * dims.size_k * dims.size_n
  rhs_bytes = rhs_size * rhs_bits // 8
  if cfgs.rhs_cfgs.has_scale:
    num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
    rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
  if cfgs.rhs_cfgs.has_bias:
    rhs_bytes += dims.size_group * dims.size_n * fp32_bytes

  out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize

  total_bytes = lhs_bytes + rhs_bytes + out_bytes

  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=total_bytes,
      transcendentals=0,
  )


def get_scope_name(cfgs: GmmConfigs) -> str:
  dims = cfgs.dims
  tiles = cfgs.tiles
  return (
      f"gmm_v2-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
      f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
  )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None = None,
):
  """Fills the GMM config for the GMM kernel."""

  dims = validate_inputs(
      lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, fuse_act
  )

  if rhs_scale is not None:
    has_scale = True
    rhs_quant_dtype = rhs.dtype
    num_blocks = rhs_scale.shape[1]
    block_size = dims.size_k // num_blocks
  else:
    has_scale = False
    rhs_quant_dtype = None
    block_size = dims.size_k

  rhs_cfgs = InputConfigs(
      quant_dtype=rhs_quant_dtype,
      quant_block_size=block_size,
      dtype=rhs.dtype,
      has_bias=rhs_bias is not None,
      has_scale=has_scale,
  )

  lhs_q_dtype = None
  if maybe_quantize_lhs and rhs_cfgs.should_dequantize_after_matmul:
    # Choose lhs quantization dtype based on TPU hardware support.
    is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
    tpu_info = pltpu.get_tpu_info()
    # Check if there is hardware compute support for rhs dtype group.
    if tpu_info.fp8_ops_per_second > 0:
      # Special handling for 4-bit integer rhs as it can be converted to fp8
      # without a numeric issues. Note that this is not the case for 4-bit
      # floating rhs as conversion to int8 will cause numeric issues.
      is_rhs_4bits = jax.dtypes.itemsize_bits(rhs_quant_dtype) == 4
      if is_rhs_float or is_rhs_4bits:
        lhs_q_dtype = jnp.float8_e4m3fn.dtype
    if tpu_info.int8_ops_per_second > 0:
      if not is_rhs_float:
        lhs_q_dtype = jnp.int8.dtype

  lhs_cfgs = InputConfigs(
      quant_dtype=lhs_q_dtype,
      # Input quantization involves reading all elements in a block to compute
      # scale value. Since this operation is very memory intensive, we use a
      # block size that is small enough to minimize memory overhead but large
      # enough to minimize compute overhead of quantization.
      quant_block_size=512,
      dtype=lhs.dtype,
  )

  if out_dtype is None:
    out_dtype = lhs.dtype

  if acc_dtype is None:
    if lhs_cfgs.quant_dtype is None:
      acc_dtype = jnp.float32.dtype
    else:
      # Input quantization requires elementwise ops which can put pressure on
      # VPUs. Using faster bf16 hardware during accumulation can help offset the
      # pressure.
      acc_dtype = jnp.bfloat16.dtype

  if isinstance(tile_info, TileSizes):
    tiles = tile_info
  else:
    tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

  return GmmConfigs(
      dims=dims,
      tiles=tiles,
      lhs_cfgs=lhs_cfgs,
      rhs_cfgs=rhs_cfgs,
      out_dtype=jnp.dtype(out_dtype),
      acc_dtype=jnp.dtype(acc_dtype),
      zero_init=zero_initialize,
      fuse_act=fuse_act,
  )


def get_metadata(cfgs: GmmConfigs) -> dict[str, str | int | float]:
  cfgs_dict = dataclasses.asdict(cfgs)
  ret = {}
  for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
    key = jax.tree_util.keystr(path, simple=True, separator=".")
    if not isinstance(val, str | int | float):
      val = str(val)
    ret[key] = val
  return ret


# K1: the fork exposes the UNJITTED body as gmm_v2_unjitted so check_k1.py can
# call it per segment inside shard_map (a @jax.jit inside shard_map is a known
# 10x+ Pallas compile-time trap). gmm_v2 below keeps the original jitted API.
def gmm_v2_unjitted(
    lhs: jax.Array,  # [size_m, size_k]
    rhs: jax.Array,  # [size_group, size_k, size_n]
    group_sizes: jax.Array,  # int32[size_lhs_group]
    rhs_scale: jax.Array | None = None,  # [size_group, num_blocks, 1, out_size]
    rhs_bias: jax.Array | None = None,  # [size_group, 1, out_size]
    group_offset: jax.Array | None = None,  # int32[1]
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
) -> jax.Array:
  """GMM kernel implemented with emit_pipeline.

  Dynamically calculate offset lhs/out tiles to reduce redundant computations.
  Additionally, it adjusts dma size based on number of valid rows and utilize
  triple buffering on weights to better utilize memory.

  Args:
    lhs: lhs with shape [size_m, size_k].
    rhs: rhs with shape [size_group, size_k, size_n].
    group_sizes: The group sizes of lhs rows of shape [size_lhs_group,].
    rhs_scale: The rhs scale of shape [size_group, num_blocks, 1, out_size].
    rhs_bias: The rhs bias of shape [size_group, 1, out_size].
    group_offset: Optional. The group offset of shape [1,].
    tile_info: The tile sizes or tile function to use.
    vmem_limit_bytes: Optional vmem limit in bytes.
    precision: Unused. Exists for compatibility reasons.
    preferred_element_type: Optional jnp.dtype for the output matrix.
    acc_dtype: Optional jnp.dtype for the accumulator.
    maybe_quantize_lhs: Quantize lhs if set to True and rhs is quantized.
    zero_initialize: Whether to initialize unvisited output elements to zero.
    fuse_act: Activation function to fuse with GMM, None if no fusion.

  Returns:
    Output of shape [size_m, size_n].
  """

  del precision

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  cfgs = make_gmm_configs(
      lhs,
      rhs,
      rhs_scale,
      rhs_bias,
      group_sizes,
      group_offset,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=preferred_element_type,
      acc_dtype=acc_dtype,
      maybe_quantize_lhs=maybe_quantize_lhs,
      zero_initialize=zero_initialize,
      fuse_act=fuse_act,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles

  # Prepare block specs.
  rhs_scale_spec = rhs_bias_spec = None
  if rhs_scale is not None:
    rhs_scale = rhs_scale.astype(jnp.float32)
    rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  if rhs_bias is not None:
    rhs_bias = rhs_bias.astype(jnp.float32)
    rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

  # Initialize scratch shapes.
  acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
  scratch_shapes = [
      # partial_out_ref
      pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
      # acc_ref
      pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
      # metadata_ref  (SMEM-FIX: O(num_groups), M-independent; was O(num_tiles))
      MetadataRef(
          group_start=pltpu.SMEM((dims.size_lhs_group + 1,), jnp.int32),
          gm_prefix=pltpu.SMEM((dims.size_lhs_group + 1,), jnp.int32),
          group_offset=pltpu.SMEM((1,), jnp.int32),
          bounds=pltpu.SMEM((2,), jnp.int32),
      ),
  ]

  num_lanes = pltpu.get_tpu_info().num_lanes
  if cfgs.zero_init:
    # TODO: Create better heuristics for determining this value.
    target_zero_ref_bytes = 32 * 1024

    # Zero initialization is done by tiling size_m dim where each tile invokes
    # zero initializing DMA for up-to tile_zero_m rows. This means larger
    # tile_zero_m will result in fewer number of tiles and lead to smaller
    # overhead. However, in order to invoke DMA call up-to tile_zero_m rows, we
    # need to store equivalent sized memory in VMEM buffer for the duration of
    # DMA. Storing [tile_zero_m, size_n] in buffer will trigger OOM if
    # tile_zero_m is too large. Instead, if we set column size as num_lanes
    # (which is smallest allowed column size for DMA) and reuse the buffer by
    # size_n//num_lanes times in a single tile, we can significantly increase
    # tile_zero_m without triggering OOM.
    out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
    tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
    tile_zero_m = min(tile_zero_m, dims.size_m)

    scratch_shapes += [
        pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
        pltpu.SemaphoreType.DMA((1,)),
    ]
  else:
    scratch_shapes += [None, None]

  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
  rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

  return pl.pallas_call(
      functools.partial(kernel_main, cfgs=cfgs),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              WeightsRef(
                  weight=pl.BlockSpec(memory_space=pltpu.HBM),
                  scale=rhs_scale_spec,
                  bias=rhs_bias_spec,
              ),
          ],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=get_scope_name(cfgs),
      cost_estimate=get_cost_estimate(cfgs),
      metadata=get_metadata(cfgs),
  )(group_sizes, group_offset, lhs, rhs_weights)[:, : cfgs.out_size_n]


gmm_v2 = jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "maybe_quantize_lhs",
        "zero_initialize",
        "fuse_act",
    ]
)(gmm_v2_unjitted)


# =============================================================================
# K1: fused a2a-in-kernel + gmm dispatch (K1_KERNEL_SPEC.md / WIRE_FORMAT.md v0)
# =============================================================================

DEVICE_MESH = pl.DeviceIdType.MESH


def make_device_id_fn(ep_axis_name=None, mesh_axis_names=None, mesh_shape=None):
  """Build a `rank -> device_id` tuple maker for `DeviceIdType.MESH`.

  Default (both None): single-axis "ep" behaviour `(rank,)` — exactly what
  check_k1/check_k2 use under a 1-axis ("ep",) mesh. MULTI-AXIS integration
  (MaxText's full mesh: data/fsdp/expert/tensor/...): pass the ambient shard_map
  mesh's axis_names + shape and the EP axis name; the maker returns the FULL
  mesh-coordinate tuple in mesh_axis_names ORDER with the ep slot = rank and
  every other axis pinned to its current `lax.axis_index` (0 for size-1 axes).
  This mirrors moe.py::_direct_reduce_scatter._mesh_device_id — the length-1
  `(rank,)` tuple is WRONG (targets the wrong device) under a rank>1 mesh.
  `rank` may be a static python int (barrier signals) or a traced int32 (sends).
  """
  if ep_axis_name is None or mesh_axis_names is None:
    return lambda rank: (rank,)

  def _fn(rank):
    return tuple(
        rank
        if nm == ep_axis_name
        else (0 if (mesh_shape is not None and mesh_shape[nm] == 1) else jax.lax.axis_index(nm))
        for nm in mesh_axis_names
    )

  return _fn


def k1_kernel_main(
    # Scalar prefetch (1-D SMEM, arithmetic indexing only)
    c_flat_ref: jax.Array,  # int32[EP*EP*E_local] — gathered counts C (§2),
    # C[src, dst, e] at index (src*EP + dst)*E_local + e
    my_id_ref: jax.Array,  # int32[1] — this device's EP rank
    # In
    x_send_ref: jax.Array,  # [M + EP*BLK, K] HBM — send-order rows (§1),
    # SEGMENT-ALIGNED: dst segment d starts at sum_{d'<d} align_up(rows_d',
    # BLK) (gmm_a2a's XLA-level repack; every DMA offset becomes a tile-row
    # multiple — Mosaic cannot slice tiled HBM at ragged row offsets)
    rhs_ref: WeightsRef,  # weight [E_local, K, N] HBM
    # PHASE E (cfgs.lhs_row_scale only): scale_send_ref [M_al, num_lanes] f32
    # HBM input follows rhs_ref — the per-row dequant scales in the SAME
    # BLK-aligned segment layout as x_send, each row's scale replicated across
    # the lane row (see gmm_a2a's fp8 notes).
    # Out
    #   y_out_ref: [EP*CAP, aligned_n] HBM — slot layout (§3), flat
    #   x_recv_ref: [EP*CAP, K] HBM — recv slots, flat; slot src occupies rows
    #     [src*CAP, (src+1)*CAP); written by REMOTE DMAs from peers. Declared
    #     as a (discarded) pallas OUTPUT: Mosaic cannot allocate HBM scratch
    #     and cannot infer ANY-space scratch here — an XLA output buffer is
    #     the proven splash_wag receive-buffer pattern.
    #   scale_recv_ref (PHASE E, cfgs.lhs_row_scale only): [EP*CAP, num_lanes]
    #     f32 — the sidecar's recv slots, same pattern as x_recv.
    # Scratch
    #   partial_out_ref: [size_lhs_sublane, tile_n] VMEM (shared, 4x)
    #   acc_ref: [tile_m, tile_n] VMEM (shared, 4x)
    #   metadata_ref: MetadataRef — SMEM, refilled per src segment
    #   zero_ref: [tile_zero_m, num_lanes] VMEM
    #   zero_sem_ref: DMA sem [1] for the zero-fill copies
    #   send_sem: DMA sem [EP], slot dst — signaled as my sends land
    #   recv_sem: DMA sem [EP], slot = SENDER's rank
    *rest,
    cfgs: GmmConfigs,  # PER-SEGMENT configs: dims.size_m == CAP,
    # dims.size_group == dims.size_lhs_group == E_local
    ep: int,
    cap_rows: int,
    blk: int,
    device_id_fn=None,  # rank -> DeviceIdType.MESH tuple (make_device_id_fn);
    # None => single-axis "ep" (rank,). Multi-axis meshes MUST pass a real one.
    mode: str,  # GATE3: "full" | "nosend" | "sendonly" | "nowait" — STATIC
    # python flag (each value is a distinct trace; NEVER a traced conditional
    # around emit_pipeline or the DMA loops). "nosend": skip barrier + sends +
    # recv waits + drains, run the EP pipelines on whatever is in x_recv
    # (GARBAGE — perf instrument only: gmm + kernel overheads). "sendonly":
    # barrier + sends + recv/send drains, skip all EP pipelines (output
    # GARBAGE — pure transport). "nowait" (OVERLAP PROBE): barrier + sends
    # fire-and-forget, run ALL pipelines immediately with NO recv waits
    # (x_recv contents racy/GARBAGE), drain send AND recv sems only at the
    # very end — measures max(compute, transport) if the engines overlap,
    # compute + transport if they serialize. Only "full" is checked for
    # correctness (check_k1.py).
    #
    # SELF-SEGMENT DIRECT READ (all modes): the own segment (src == my) is
    # never DMA'd — its send is skipped (pl.when(dst != my); my is traced) and
    # its pipeline (statically src_pos == 0) reads lhs straight from the
    # BLK-aligned x_send buffer at my aligned segment start (same rows, same
    # order the self-DMA would have delivered -> bit-exact). No recv wait for
    # src==my; send/recv byte accounting excludes the self segment on BOTH
    # sides symmetrically (every device skips its own); x_recv slot [my]
    # simply stays unused.
):
  """K1 body: barrier -> blocked remote sends -> EP gated pipelines -> drain.

  Single grid-less pallas_call; the EP per-src emit_pipelines are a STATIC
  python loop (Mosaic rejects emit_pipeline under lax control flow). Processing
  order is (my + src_pos) % EP — own segment first (§5), so the local gmm hides
  the first remote segment's flight time.
  """
  # PHASE E: the fp8 path adds a scale-send input (after rhs) and a scale-recv
  # output (after x_recv); unpack positionally on the static flag.
  if cfgs.lhs_row_scale:
    (
        scale_send_ref,
        y_out_ref,
        x_recv_ref,
        scale_recv_ref,
        partial_out_ref,
        acc_ref,
        metadata_ref,
        zero_ref,
        zero_sem_ref,
        send_sem,
        recv_sem,
    ) = rest
  else:
    scale_send_ref = scale_recv_ref = None
    (
        y_out_ref,
        x_recv_ref,
        partial_out_ref,
        acc_ref,
        metadata_ref,
        zero_ref,
        zero_sem_ref,
        send_sem,
        recv_sem,
    ) = rest

  e_local = cfgs.dims.size_lhs_group
  size_k = cfgs.dims.size_k
  sublane = cfgs.dims.size_lhs_sublane
  num_k = pl.cdiv(size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

  my = my_id_ref[0]
  if device_id_fn is None:
    device_id_fn = lambda r: (r,)  # single-axis "ep" default (check_k1/k2)

  # Keep the fp8/packed plumbing intact (no-op for bf16), as in kernel_main.
  if cfgs.rhs_cfgs.should_bitcast:
    rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
    rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

  # All a2a DMAs run on 3-D SUBLANE VIEWS (rows grouped in tile-row units, the
  # zero_out idiom): the leading dim of a (rows/sublane, sublane, K) HBM view
  # carries no tile-alignment constraint, so traced tile-row offsets are always
  # legal. Segment starts are tile-row multiples by construction (the
  # XLA-level BLK-aligned repack in gmm_a2a).
  ch3 = blk // sublane  # DMA block size in tile-row units (static)
  x_send3 = x_send_ref.reshape(-1, sublane, size_k)
  x_recv3 = x_recv_ref.reshape(-1, sublane, size_k)
  # PHASE E: the scale sidecar buffers share the row layout (same sublane
  # grouping, same BLK-aligned segment starts), so every DMA below reuses the
  # rows' tile-row offsets verbatim — one extra copy per (dst, block), same
  # send/recv semaphores.
  if cfgs.lhs_row_scale:
    scale_w = scale_send_ref.shape[-1]  # num_lanes (each scale replicated)
    scale_send3 = scale_send_ref.reshape(-1, sublane, scale_w)
    scale_recv3 = scale_recv_ref.reshape(-1, sublane, scale_w)

  # ---- 1. Barrier (once): no peer may still be reading its x_recv from the
  # previous invocation when our sends start overwriting it. (Skipped entirely
  # in "nosend" mode — no remote traffic exists to fence.)
  if mode != "nosend":
    bsem = pltpu.get_barrier_semaphore()
    for d in range(ep):
      pl.semaphore_signal(
          bsem, inc=1, device_id=device_id_fn(d), device_id_type=DEVICE_MESH
      )
    pl.semaphore_wait(bsem, ep)

  # ---- counts arithmetic (all scalars from the 1-D SMEM table)
  def _seg_rows(base):
    """sum_e c_flat[base + e] — rows in one (src, dst) segment."""
    return lax.fori_loop(
        0, e_local, lambda e, a: a + c_flat_ref[base + e], jnp.int32(0)
    )

  # ---- 2. Sends (GATE3 redo: ROUND-ROBIN, block-major). Precompute every
  # dst's aligned segment start (tile-row units) and block count from C, then
  # run ONE fori_loop over block index i in [0, max_nblk) whose body statically
  # unrolls over ALL EP dsts and issues block i for dst under
  # pl.when(i < nblk[dst]). Consecutive DMAs target DIFFERENT destinations, so
  # both ICI link directions (+ intra-hop routes) run concurrently instead of
  # draining one peer at a time (the v0 per-dst-sequential transport suspect).
  # Semaphore/slot discipline is UNCHANGED: send_sem slot = dst, recv_sem slot
  # = SENDER's rank, same blocks with the same sems — only the issue ORDER
  # changes, so the byte-count gating and bit-exactness are unaffected.
  send_nblk = []
  send_seg_start3 = []  # per-dst aligned segment start, tile-row units
  send_start3 = jnp.int32(0)
  my_slot3 = my * (cap_rows // sublane)
  for dst in range(ep):
    rows = _seg_rows((my * ep + dst) * e_local)
    nblk = pl.cdiv(rows, blk)  # traced; tail block carries slack rows
    send_nblk.append(nblk)
    send_seg_start3.append(send_start3)
    send_start3 = send_start3 + nblk * ch3  # next aligned segment start

  # SELF-SEGMENT: my own aligned start (tile-row units) — the src==my pipeline
  # reads its lhs from x_send3 here instead of x_recv slot my. `my` is traced,
  # so select it from the static per-dst list (int32 arithmetic, no gather).
  my_seg_start3 = jnp.int32(0)
  for dst in range(ep):
    my_seg_start3 = jnp.where(my == dst, send_seg_start3[dst], my_seg_start3)

  if mode != "nosend":
    # PER-DST SEQUENTIAL issue in ring order (my+1 entirely first, then my+2, ...).
    # Round-robin-across-dsts made every peer's segment complete together at the
    # transport tail, so segments 2..EP all gated at full transport time (measured:
    # fused 24.6ms vs nowait 19.5ms, gap = 3 late waits). Ring order staircases
    # arrival of segment my+d at ~d/(EP-1) of transport, each covered by the
    # preceding segments' compute. d >= 1 => dst != my structurally (the own
    # segment is read directly from x_send3 by the src_pos == 0 pipeline).
    def _select_traced(lst, idx):
      v = lst[0]
      for j in range(1, ep):
        v = jnp.where(idx == j, lst[j], v)
      return v

    for d in range(1, ep):  # static ring offset; dst is traced
      dst = jnp.mod(my + jnp.int32(d), jnp.int32(ep))
      nblk_d = _select_traced(send_nblk, dst)
      start3_d = _select_traced(send_seg_start3, dst)

      def _send_blocks(i, carry, dst=dst, start3_d=start3_d):
        pltpu.make_async_remote_copy(
            x_send3.at[pl.ds(start3_d + i * ch3, ch3)],
            x_recv3.at[pl.ds(my_slot3 + i * ch3, ch3)],
            send_sem.at[dst],
            recv_sem.at[my],
            device_id=device_id_fn(dst),
            device_id_type=DEVICE_MESH,
        ).start()
        # PHASE E sidecar: this block's per-row scales, SAME offsets (the
        # sidecar buffers mirror the row layout), SAME send/recv semaphores —
        # the drain byte accounting includes both (see _drain_dma_sem).
        if cfgs.lhs_row_scale:
          pltpu.make_async_remote_copy(
              scale_send3.at[pl.ds(start3_d + i * ch3, ch3)],
              scale_recv3.at[pl.ds(my_slot3 + i * ch3, ch3)],
              send_sem.at[dst],
              recv_sem.at[my],
              device_id=device_id_fn(dst),
              device_id_type=DEVICE_MESH,
          ).start()
        return carry

      lax.fori_loop(0, nblk_d, _send_blocks, jnp.int32(0))

  # Drain-by-bytes for DMA semaphores: jax 0.10.1 rejects pl.semaphore_wait on
  # a dma_sem ("Use pl.semaphore_wait" is only for REGULAR/BARRIER; DMA sems
  # need the dma_wait primitive). Reconstruct a local copy descriptor whose
  # slice has exactly the signaled byte count (traced size — the zero_out_end
  # idiom) and .wait() it: byte-identical to the splash_wag drain, and the copy
  # handles still never cross scopes.
  def _drain_dma_sem(sem_view, nblocks):
    rows3 = nblocks * ch3  # bytes = nblocks * BLK * K * itemsize
    pltpu.make_async_copy(
        x_recv3.at[pl.ds(0, rows3)],  # refs only size the wait
        x_recv3.at[pl.ds(0, rows3)],
        sem_view,
    ).wait()
    # PHASE E: the sidecar DMAs signal the SAME semaphores — drain their byte
    # count too (a second reconstructed copy, sized on the scale view), on
    # BOTH the send and recv drains, symmetrically (the self segment sends
    # neither rows nor scales, so nblocks=0 covers it on both).
    if cfgs.lhs_row_scale:
      pltpu.make_async_copy(
          scale_recv3.at[pl.ds(0, rows3)],
          scale_recv3.at[pl.ds(0, rows3)],
          sem_view,
      ).wait()

  # ---- 3. Per-src compute: static unroll, own segment first. Each pipeline is
  # gated by draining recv_sem[src] by the byte count the sender signaled (both
  # sides derive it from the same C table).
  lhs_in = x_recv_ref.reshape(-1, sublane, size_k)
  out_in = y_out_ref.reshape(-1, sublane, y_out_ref.shape[-1])
  scratches = [partial_out_ref, acc_ref, metadata_ref]

  for src_pos in range(ep):
    # CONSUME ORDER mirrors the SEND ORDER: sender s ships to (s+d) at stage d,
    # so receiver r's data arrives from (r-1) first, (r-2) second, ... Processing
    # (my + src_pos) instead anti-correlates with arrivals — the first remote
    # segment waited on (my+1) is the LAST one sent to us (measured: 24.6ms vs
    # the 19.5ms nowait ideal, gap = late gates). src_pos == 0 stays the self
    # segment either way.
    src = (my - src_pos + ep) % ep  # traced
    base_c = (src * ep + my) * e_local
    # Recv wait: skipped for src_pos == 0 (STATICALLY the self segment —
    # src == my — which is never sent) and in the no-wait modes ("nosend":
    # nothing was sent; "nowait": fire-and-forget, sems drained at the end).
    if mode not in ("nosend", "nowait") and src_pos != 0:
      recv_rows = _seg_rows(base_c)
      nblk_recv = pl.cdiv(recv_rows, blk)
      _drain_dma_sem(recv_sem.at[src], nblk_recv)
    if mode == "sendonly":  # transport only: no metadata/zero/pipeline
      continue

    # Segment metadata: group sizes = C[src, my, :] (base offset into c_flat).
    num_gm = fill_metadata(
        c_flat_ref, None, metadata_ref, cfgs=cfgs, base=base_c
    )

    row_base = src * (cap_rows // sublane)  # traced, sublane units
    if cfgs.zero_init:
      zero_size = zero_out_start(
          y_out_ref,
          zero_ref,
          zero_sem_ref,
          metadata_ref,
          num_gm,
          cfgs=cfgs,
          row_offset=src * cap_rows,
          num_rows=cap_rows,
      )

    # SELF-SEGMENT direct read: src_pos == 0 is statically src == my — its lhs
    # comes straight from the BLK-aligned x_send buffer at my aligned segment
    # start (same rows/order the skipped self-DMA would have delivered); the
    # output row_base is unchanged (y_out slot my). Remote segments read their
    # x_recv slot as before.
    if src_pos == 0:
      lhs_buf, lhs_row_base = x_send3, my_seg_start3
    else:
      lhs_buf, lhs_row_base = lhs_in, None  # None -> row_base (x_recv slot)
    (lhs_spec, rhs_spec), out_spec = generate_block_specs(
        metadata_ref, cfgs, row_base=row_base, lhs_row_base=lhs_row_base
    )
    if cfgs.lhs_row_scale:
      # PHASE E: the scale sidecar is a third pipeline input, tile-fetched by
      # the SAME row indexing as the lhs (self segment: send-side buffer at my
      # aligned start; remote: the recv slot base — identical unit semantics
      # because both buffers share the sublane grouping).
      if src_pos == 0:
        scale_buf, scale_row_base = scale_send3, my_seg_start3
      else:
        scale_buf, scale_row_base = scale_recv3, row_base
      scale_spec = make_lhs_scale_spec(metadata_ref, cfgs, scale_row_base)
      pipeline_fn = pltpu.emit_pipeline(
          functools.partial(inner_kernel_rs, cfgs=cfgs),
          grid=(num_n, num_gm, num_k),
          in_specs=(lhs_spec, rhs_spec, scale_spec),
          out_specs=out_spec,
      )
      pipeline_fn(lhs_buf, rhs_ref, scale_buf, out_in, scratches=scratches)
    else:
      pipeline_fn = pltpu.emit_pipeline(
          functools.partial(inner_kernel, cfgs=cfgs),
          grid=(num_n, num_gm, num_k),
          in_specs=(lhs_spec, rhs_spec),
          out_specs=out_spec,
      )
      pipeline_fn(lhs_buf, rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
      zero_out_end(y_out_ref, zero_sem_ref, zero_size, dims=cfgs.dims)

  # ---- 4. Drain sends: don't retire with in-flight outbound DMAs. The self
  # segment was never sent, so send_sem[my] holds 0 signals — drain 0 blocks
  # there (traced-size zero wait, the zero_out_end idiom).
  if mode != "nosend":
    for dst in range(ep):
      nblk_send = jnp.where(my == dst, jnp.int32(0), send_nblk[dst])
      _drain_dma_sem(send_sem.at[dst], nblk_send)

  # ---- 5. "nowait" only: the recv sems were never waited on before the
  # pipelines; drain them now so the kernel retires with clean semaphores and
  # no in-flight inbound DMAs (this final wait is exactly what makes nowait
  # measure max(compute, transport)). Self segment (src_pos == 0) excluded —
  # never sent.
  if mode == "nowait":
    for src_pos in range(1, ep):
      src = (my + src_pos) % ep  # traced
      recv_rows = _seg_rows((src * ep + my) * e_local)
      _drain_dma_sem(recv_sem.at[src], pl.cdiv(recv_rows, blk))


def _k1_cost_estimate(cfgs: GmmConfigs, ep: int) -> pl.CostEstimate:
  """Upper-bound cost: EP segments of up to CAP rows each + the a2a bytes."""
  dims = cfgs.dims
  flops = 2 * ep * dims.size_m * dims.size_k * dims.size_n
  lhs_bytes = ep * dims.size_m * dims.size_k * jnp.dtype(cfgs.lhs_cfgs.dtype).itemsize
  rhs_bytes = dims.size_group * dims.size_k * dims.size_n * jnp.dtype(
      cfgs.rhs_cfgs.dtype
  ).itemsize
  out_bytes = ep * dims.size_m * cfgs.out_size_n * jnp.dtype(cfgs.out_dtype).itemsize
  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=2 * lhs_bytes + rhs_bytes + out_bytes,  # a2a + gmm read
      transcendentals=0,
  )


def k1_make_cfgs(
    lhs_dtype,
    rhs_dtype,
    size_k: int,
    size_n: int,
    e_local: int,
    cap_rows: int,
    *,
    blk: int = 512,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    rhs_scale: jax.ShapeDtypeStruct | None = None,  # PHASE E: fp8 weights'
    # per-block scale [E_local, num_blocks, 1, N] (shape/dtype only)
    maybe_quantize_lhs: bool = True,  # PHASE E: False on the fp8-wire path
    # (the lhs arrives ALREADY quantized; re-quantizing it in-kernel would be
    # wrong). Inert without rhs_scale, so the bf16 default is unchanged.
    out_dtype: jnp.dtype | None = None,  # PHASE E: fp8-wire callers must
    # override (None -> lhs dtype, which would be fp8)
    lhs_row_scale: bool = False,  # PHASE E: per-row dequant sidecar
) -> Tuple[GmmConfigs, int]:
  """Per-SEGMENT GmmConfigs for K1 (dims.size_m = CAP) + the pallas vmem limit.

  Identical to what gmm_v2 would derive for one [CAP, K] x [E_local, K, N]
  call — the bit-exactness anchor. check_k1.py hands the reference gmm_v2
  `k1_make_cfgs(...)[0].tiles` as tile_info: a different tile_k would change
  the split-K f32 accumulation order and break bit-exactness.
  """
  del blk  # kept in the signature for forward-compat of the tiling budget
  if vmem_limit_bytes is None:
    vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

  cfgs = make_gmm_configs(
      jax.ShapeDtypeStruct((cap_rows, size_k), lhs_dtype),
      jax.ShapeDtypeStruct((e_local, size_k, size_n), rhs_dtype),
      rhs_scale,
      None,  # rhs_bias
      jax.ShapeDtypeStruct((e_local,), jnp.int32),
      jax.ShapeDtypeStruct((1,), jnp.int32),
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      out_dtype=out_dtype,
      acc_dtype=None,
      maybe_quantize_lhs=maybe_quantize_lhs,
      zero_initialize=True,  # v0 keeps zero_init — correctness first
      fuse_act=None,
  )
  if lhs_row_scale:
    cfgs = dataclasses.replace(cfgs, lhs_row_scale=True)
  return cfgs, vmem_limit_bytes


def gmm_a2a(
    x_send: jax.Array,  # [M, K] bf16 — rows in send order (WIRE_FORMAT §1).
    # PHASE E: may be float8_e4m3fn (the wire dtype IS the MXU dtype) when
    # lhs_row_scale is provided.
    counts_c: jax.Array,  # int32[EP, EP, E_local] — gathered counts table C (§2)
    rhs: jax.Array,  # [E_local, K, N] bf16 — local expert weights.
    # PHASE E: may be float8_e4m3fn with rhs_scale (the fork's machinery).
    *,
    my_id: jax.Array,  # int32 scalar — this device's EP rank (lax.axis_index)
    ep: int,
    cap_rows: int,  # per-src slot rows; v0: CAP = M (dropless worst case)
    collective_id: int = 42,
    blk: int = 512,  # rows per DMA block (static slice size; starts are traced)
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    mode: str = "full",  # GATE3: "full" | "nosend" | "sendonly" | "nowait".
    # STATIC python flag — each value produces its own traced/compiled variant
    # (never a traced conditional around emit_pipeline or the DMA loops).
    # "nosend", "sendonly" and "nowait" outputs are GARBAGE (perf
    # decomposition instruments only); only "full" is checked for correctness.
    # "nowait" (OVERLAP PROBE spec): sends fire-and-forget, all pipelines run
    # with no recv waits, send+recv sems drained at the very end —
    # discriminates engine overlap (≈ max(compute, transport)) from true
    # queue serialization (≈ compute + transport).
    prealigned: bool = False,  # PHASE D: x_send is ALREADY in the BLK-aligned
    # segment layout ([M + EP*BLK, K]; dst segment d starts at
    # sum_{d'<d} align_up(rows_d', BLK)) — SKIP the internal XLA repack.
    # Used by the backward: dY_home comes out of the combine-bwd expansion
    # already in this layout (it IS K1's x_send layout, see gmm_return's
    # y_home contract). The kernel walks the same C-derived aligned starts
    # either way; only the wrapper-level jnp.take pass is skipped. The
    # aligned length is asserted statically (M + EP*BLK); the per-segment
    # starts are data-dependent (derived from C in-kernel) and cannot be
    # statically checked.
    return_recv: bool = False,  # PHASE D: also return the x_recv receive
    # buffer, reshaped [EP, CAP, K]. x_recv is ALREADY a pallas output (the
    # Mosaic-can't-allocate-HBM-scratch pattern) — this flag only stops the
    # wrapper from discarding it. NOTE the self slot [my] is NEVER written
    # (self-segment direct read); callers needing it (the vjp residual /
    # bwd dz) must patch slot my from the send buffer at MY aligned start.
    # Wrapper-level only — no kernel change.
    lhs_row_scale: jax.Array | None = None,  # PHASE E: f32[M] — per-token
    # dequant scales (amax/448, zero-amax -> 1), produced by the home-side
    # quantizer (quant_utils.quantize_rows on the send-order rows, so they
    # are PRE-PERMUTED into send order). Setting this selects the fp8
    # dispatch path: x_send must be float8_e4m3fn; the scales ride the SAME
    # blocked-DMA structure as the rows (one sidecar DMA per (dst, block),
    # same send/recv semaphores, byte accounting includes both, symmetric
    # minus the self segment); the kernel epilogue multiplies the f32
    # accumulator by the row's scale before the out-dtype cast.
    # SIDECAR LAYOUT (documented deviation from PHASE_E_SPEC's "BLK x 4 B"):
    # each scale is replicated across a full num_lanes row ([M_al, 128] f32,
    # BLK x 512 B per block DMA, ~7% of the fp8 row bytes at K=7168). Mosaic's
    # only proven per-row-scalar broadcast is a keepdims LANE-REDUCE (how
    # block_scale works — the header law forbids [:, None]); a compact 4 B/row
    # sidecar can neither be tile-fetched per m-tile (fetch granularity is a
    # (sublane, 128) f32 tile = 1024 scales vs 32-row-aligned m-tiles) nor
    # lane->sublane transposed on the TC path. v1 candidates: in-kernel
    # relayout, or packing scales into spare row bytes.
    rhs_scale: jax.Array | None = None,  # PHASE E: [E_local, num_blocks, 1, N]
    # f32 — fp8 weight per-K-block scales (quant_utils.quantize_weights).
    # quant_block = K/num_blocks must be >= mxu_column_size (256) or the
    # kernel dequantizes BEFORE the matmul = silent bf16 cliff (asserted).
    preferred_element_type: jnp.dtype | None = None,  # PHASE E: out dtype.
    # None -> bf16 on the fp8 wire (spec v0: output stays bf16; check_fp8's
    # plumbing gate passes f32 to compare at f32 fidelity), lhs dtype else.
    ep_axis_name: str | None = None,  # MULTI-AXIS mesh: the EP axis name +
    mesh_axis_names: tuple | None = None,  # the shard_map mesh axis_names +
    mesh_shape=None,  # its shape. All None => single-axis "ep" device_id=(rank,)
    # (check_k1/k2). Under MaxText's full mesh these MUST be set or the in-kernel
    # remote DMAs target the wrong device (see make_device_id_fn).
    interpret: bool = False,
) -> jax.Array:  # [EP, CAP, N] — outputs in slot layout (§3); rows beyond
  # C[src, me].sum() are zero (zero_init runs per segment)
  # (return_recv=True: a (y_out, x_recv) tuple — see the flag note above)
  """Fused dispatch: in-kernel a2a (remote DMAs) + chunk-by-src gmm (§5).

  Call under shard_map over the `ep` axis (check_rep/check_vma False). The
  receive buffer x_recv [EP*CAP, K] is an internal HBM scratch of the
  pallas_call, written by remote DMAs from the peers; y_out is the pallas
  output. Bit-exactness target: prep_a2a.reference_layer's per-src segment
  slicing with THIS file's gmm_v2 per segment (see check_k1.py).
  """
  assert mode in ("full", "nosend", "sendonly", "nowait"), mode
  size_m, size_k = x_send.shape
  if prealigned:
    # x_send is the aligned buffer [M + EP*BLK, K]; recover the ragged M for
    # the layout asserts below (sum(align_up(rows_d, BLK)) <= M + EP*BLK is
    # the static bound the kernel's C-derived walk relies on).
    size_m = size_m - ep * blk
    assert size_m > 0, (x_send.shape, ep, blk)
  e_local, size_k2, size_n = rhs.shape
  assert size_k2 == size_k
  assert counts_c.shape == (ep, ep, e_local), counts_c.shape
  assert counts_c.dtype == jnp.int32

  # ---- PHASE E: fp8 dispatch path guards ----
  fp8_wire = lhs_row_scale is not None
  if fp8_wire:
    assert x_send.dtype == jnp.float8_e4m3fn, (
        "lhs_row_scale set but x_send is not float8_e4m3fn", x_send.dtype
    )
    assert not prealigned, (
        "fp8 + prealigned: the prealigned path is the BACKWARD's entry and "
        "fp8 bwd is OUT OF SCOPE in Phase E v0 (the vjp calls the bf16 kernel "
        "paths). A prealigned fp8 dispatch would need the caller to hand the "
        "scale sidecar in the aligned layout too — not built."
    )
    assert lhs_row_scale.shape == (size_m,), (lhs_row_scale.shape, size_m)
    assert lhs_row_scale.dtype == jnp.float32, lhs_row_scale.dtype
  if rhs_scale is not None:
    assert rhs_scale.shape[0] == e_local and rhs_scale.shape[2:] == (1, size_n)
  if preferred_element_type is None and fp8_wire:
    preferred_element_type = jnp.bfloat16  # spec v0: output stays bf16

  # GATE3: production-tile default. gmm_iso measured TileSizes(256, 7168,
  # 1024) (maxtext wi_tile_fwd) at 3.76 ms/iter vs 6.3 ms/iter for
  # calculate_tiling on the production K=7168 segment shape — make it the
  # default there. Simple guard: an EXPLICIT tile_info always wins; every
  # other K keeps calculate_tiling.
  # fp8-AWARE (2026-07): fp8 halves operand bytes -> a bigger tile_n fits VMEM,
  # and bench_fp8_fair measured fp8-best = (256, 7168, 2048) vs bf16-best (256,
  # 7168, 1024). The bf16 tile was crippling the fp8 path (forced tn=1024).
  if tile_info is calculate_tiling and size_k == 7168:
    tile_info = (
        TileSizes(tile_m=256, tile_k=7168, tile_n=2048)
        if fp8_wire
        else TileSizes(tile_m=256, tile_k=7168, tile_n=1024)
    )
  # Blocked-DMA layout guarantees (spec step 2): the tail block of a segment
  # may carry up to BLK-1 slack rows, and writes land in a CAP-row slot — both
  # require BLK alignment; the repack below gives x_send EP*BLK slack rows.
  assert size_m % blk == 0, (size_m, blk)
  assert cap_rows % blk == 0, (cap_rows, blk)
  assert cap_rows >= size_m, (cap_rows, size_m)  # v0: dropless worst case
  itemsize = jnp.dtype(x_send.dtype).itemsize
  assert (blk * size_k * itemsize) % 128 == 0  # DMA granule
  assert (size_k * itemsize) % 128 == 0  # row granule

  # Per-SEGMENT gmm configs (dims.size_m = CAP). The reference gmm_v2 in
  # check_k1.py must be handed cfgs.tiles (see k1_make_cfgs) for bit-exactness.
  cfgs, vmem_limit_bytes = k1_make_cfgs(
      x_send.dtype,
      rhs.dtype,
      size_k,
      size_n,
      e_local,
      cap_rows,
      blk=blk,
      tile_info=tile_info,
      vmem_limit_bytes=vmem_limit_bytes,
      rhs_scale=None if rhs_scale is None else jax.ShapeDtypeStruct(
          rhs_scale.shape, jnp.float32
      ),
      # fp8 wire: the lhs is ALREADY quantized — never re-quantize in-kernel.
      maybe_quantize_lhs=not fp8_wire,
      out_dtype=preferred_element_type,
      lhs_row_scale=fp8_wire,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles
  assert cap_rows % dims.size_lhs_sublane == 0
  assert blk % dims.size_lhs_sublane == 0
  if fp8_wire and cfgs.rhs_cfgs.has_scale:
    # quant_block < mxu_column_size would dequantize BEFORE the matmul — the
    # measured silent-bf16 cliff (fp8-gmm session, opcode-verified). Refuse.
    assert not cfgs.rhs_cfgs.should_dequantize_before_matmul, (
        "rhs quant_block", cfgs.rhs_cfgs.quant_block_size,
        "< mxu_column_size — the matmul would silently run bf16",
    )
    # PHASE E v1: the fp8 row-scale dispatch takes the single-matmul fast path
    # (inner_kernel), which REQUIRES a full-channel rhs scale so both scales
    # commute out of the K contraction. Early, wrapper-level guard (the kernel
    # re-asserts on the path). Callers: pass block_k == K to quantize_weights.
    assert cfgs.rhs_cfgs.quant_block_size == size_k, (
        "fp8 dispatch requires a FULL-CHANNEL rhs scale (num_blocks == 1): "
        f"quant_block_size={cfgs.rhs_cfgs.quant_block_size} != K={size_k}. "
        "Quantize the weights with block_k = K.",
    )

  # ---- XLA-level BLK-aligned segment repack. Mosaic cannot DMA-slice a tiled
  # HBM memref at ragged row offsets ("Offsets along tiled dimensions must be
  # aligned to tiles"), and the sub-tile row phase of ragged data cannot be
  # moved by ANY DMA (only tile-row-granular slices are expressible; verified
  # empirically: HBM scratch is tiled too, and HBM->HBM copies unify layouts).
  # So the wrapper re-packs x_send once with one XLA row-gather: dst segment d
  # moves from its ragged start (prefix sums of C[my]) to the BLK-aligned
  # start sum_{d'<d} align_up(rows_d', BLK). The kernel walks the SAME aligned
  # starts (both sides derive them from C). Gap rows are slack: transferred by
  # tail blocks, never read by a receiver (group sizes bound all reads). The
  # production fix (prep emits aligned segments / a linear-layout wire buffer)
  # deletes this extra pass; tracked as a v0 cost.
  scale_wire = None
  if prealigned:
    # PHASE D: input already in the aligned layout — skip the repack pass.
    x_send_al = x_send
  else:
    send_sizes = jnp.take(counts_c, my_id, axis=0).sum(axis=1)  # [EP] rows to d
    seg_starts = jnp.cumsum(send_sizes) - send_sizes  # ragged starts in x_send
    al_spans = ((send_sizes + blk - 1) // blk) * blk
    al_starts = jnp.cumsum(al_spans) - al_spans  # aligned starts (kernel view)
    m_al = size_m + ep * blk  # static upper bound: sum(align_up) <= M + EP*BLK
    r = jnp.arange(m_al, dtype=jnp.int32)
    seg_of_r = jnp.zeros((m_al,), jnp.int32)
    for d in range(1, ep):  # static loop, int32 arithmetic (no bool masks kept)
      seg_of_r = seg_of_r + (r >= al_starts[d]).astype(jnp.int32)
    src_idx = r - jnp.take(al_starts, seg_of_r) + jnp.take(seg_starts, seg_of_r)
    src_idx = jnp.clip(src_idx, 0, size_m - 1)  # slack rows: harmless clamp
    x_send_al = jnp.take(x_send, src_idx, axis=0)  # [M + EP*BLK, K]
    if fp8_wire:
      # PHASE E sidecar: the scales ride the SAME aligned segment layout (the
      # same src_idx gather), each replicated across a full lane row so the
      # kernel epilogue can recover it with a keepdims lane-reduce (see the
      # lhs_row_scale arg note). Slack rows carry clamped (finite) scales.
      num_lanes_side = pltpu.get_tpu_info().num_lanes
      scale_al = jnp.take(lhs_row_scale.astype(jnp.float32), src_idx, axis=0)
      scale_wire = jnp.broadcast_to(
          scale_al[:, None], (m_al, num_lanes_side)
      )  # [M + EP*BLK, 128] f32 (XLA-level [:, None] — not the Mosaic trap)

  c_flat = counts_c.reshape(-1)
  my_id_arr = jnp.asarray(my_id, jnp.int32).reshape(1)

  num_lanes = pltpu.get_tpu_info().num_lanes
  target_zero_ref_bytes = 32 * 1024
  out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
  tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
  tile_zero_m = min(tile_zero_m, dims.size_m)

  scratch_shapes = [
      # partial_out_ref (shared sequentially across the EP pipelines)
      pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
      # acc_ref
      pltpu.VMEM((tiles.tile_m, tiles.tile_n), cfgs.acc_dtype),
      # metadata_ref — O(E_local), refilled per src segment
      MetadataRef(
          group_start=pltpu.SMEM((dims.size_lhs_group + 1,), jnp.int32),
          gm_prefix=pltpu.SMEM((dims.size_lhs_group + 1,), jnp.int32),
          group_offset=pltpu.SMEM((1,), jnp.int32),
          bounds=pltpu.SMEM((2,), jnp.int32),
      ),
      # zero_ref + its DMA semaphore
      pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
      pltpu.SemaphoreType.DMA((1,)),
      # send_sem[dst], recv_sem[sender]
      pltpu.SemaphoreType.DMA((ep,)),
      pltpu.SemaphoreType.DMA((ep,)),
  ]

  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_init = [
      jax.ShapeDtypeStruct((ep * cap_rows, aligned_n), cfgs.out_dtype),
      # x_recv: remote-DMA receive buffer as a discarded output (see kernel
      # docstring — Mosaic cannot allocate/infer HBM or ANY scratch here)
      jax.ShapeDtypeStruct((ep * cap_rows, size_k), x_send.dtype),
  ]
  rhs_weights = WeightsRef(
      weight=rhs,
      scale=None if rhs_scale is None else rhs_scale.astype(jnp.float32),
      bias=None,
  )
  in_specs = [
      pl.BlockSpec(memory_space=pltpu.HBM),
      WeightsRef(
          weight=pl.BlockSpec(memory_space=pltpu.HBM),
          scale=(
              None if rhs_scale is None
              else pl.BlockSpec(memory_space=pltpu.HBM)
          ),
          bias=None,
      ),
  ]
  operands = [c_flat, my_id_arr, x_send_al, rhs_weights]
  if fp8_wire:
    # PHASE E: scale-send input (after rhs) + scale-recv discarded output
    # (after x_recv) — the sidecar's x_send/x_recv mirror.
    in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
    operands.append(scale_wire)
    out_init.append(
        jax.ShapeDtypeStruct(
            (ep * cap_rows, scale_wire.shape[-1]), jnp.float32
        )
    )

  results = pl.pallas_call(
      functools.partial(
          k1_kernel_main, cfgs=cfgs, ep=ep, cap_rows=cap_rows, blk=blk,
          device_id_fn=make_device_id_fn(ep_axis_name, mesh_axis_names, mesh_shape),
          mode=mode,
      ),
      out_shape=tuple(out_init),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=in_specs,
          out_specs=tuple(
              pl.BlockSpec(memory_space=pltpu.HBM) for _ in out_init
          ),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          # "nosend" has no barrier (get_barrier_semaphore is never called);
          # Mosaic rejects a collective_id without a custom barrier.
          collective_id=None if mode == "nosend" else collective_id,
      ),
      # _fastfp8 fingerprint: proves in any pod trace WHICH code path ran (the
      # fast-path pod run was otherwise unverifiable — reviewer finding).
      name=f"k1_a2a_ep{ep}_{mode}_blk{blk}"
      + ("_fastfp8" if (cfgs.lhs_row_scale and cfgs.lhs_cfgs.quant_dtype is None
                        and jnp.issubdtype(cfgs.lhs_cfgs.dtype, jnp.floating)
                        and jax.dtypes.itemsize_bits(cfgs.lhs_cfgs.dtype) == 8
                        and cfgs.rhs_cfgs.should_dequantize_after_matmul) else "")
      + f"-{get_scope_name(cfgs)}",
      cost_estimate=_k1_cost_estimate(cfgs, ep),
      metadata=get_metadata(cfgs),
      interpret=interpret,
  )(*operands)
  out, x_recv = results[0], results[1]  # results[2] (fp8): scale_recv, dropped

  y_out = out[:, : cfgs.out_size_n].reshape(ep, cap_rows, cfgs.out_size_n)
  if return_recv:
    return y_out, x_recv.reshape(ep, cap_rows, size_k)
  return y_out


# =============================================================================
# WIRE-FORMAT V1 (WIRE_FORMAT_V1_SPEC.md §A): dedup dispatch. One row per
# (token, dst device) travels; the receiver re-expands INSIDE the gmm by
# replacing the lhs tile block-DMA with per-row gathers driven by the shipped
# index sidecar (the TC-legal §7 pattern; idioms from gather/gmm_fused.py).
#
# Wire streams per dst segment (all BLK-block DMAs on the same send/recv
# semaphores; drains reconstruct all three byte counts symmetrically — the
# sidecar/semaphore deadlock trap):
#   rows  x_dedup   [Ddup rows x K]      bf16, BLK-aligned dedup segments
#   side  side_int  [2 int32/expanded row] interleaved (dedup idx, w bits),
#                   v0's BLK-aligned EXPANDED segment layout
#   rev   rev_int   [topk int32/dedup slot] slot-major reverse map (consumed
#                   by K2-v1's reduce, not by K1 — it just rides this wire)
#
# LHS GATHER (bit-exactness carrier): for each (n, gm, k) tile the inner
# kernel stages the tile's sidecar window HBM->SMEM (128-aligned start +
# phase — 1-D int32 arrays can only be sliced at 128-element granularity),
# then per tile row r issues one DMA from the u32-BITCAST dedup buffer (bf16
# packs 2 rows per 32-bit sublane word, so a 1-row bf16 VMEM write is
# illegal — gather row abs//2, shift the wanted half by (abs&1)*16, whole-
# tile bitcast to bf16: gmm_fused.py's proven R4/v12 recipe). Indices are
# ALWAYS clamped to the dedup slot (recv slack is uninitialized HBM). The
# gathered tile feeds the UNCHANGED inner_kernel => y_out is bit-exact vs v0
# (same values in [m_start, m_end), same masking, same zero_init).
# =============================================================================


def make_v1_gather_inner(
    cfgs: GmmConfigs,
    x_u32,          # u32-bitcast HBM ref [rows//2, K] — gather source
    side_ref,       # int32 HBM ref (interleaved sidecar; this segment's base below)
    x_row_base,     # traced int32: first bf16 row of this segment in x_u32's source
    side_elem_base, # traced int32: first sidecar ELEMENT (2*expanded row) of segment
    clamp_rows,     # static int: dedup slot rows (index clamp bound)
):
  """emit_pipeline inner for K1-v1: gather lhs tile, then vanilla inner_kernel."""
  sublane = cfgs.dims.size_lhs_sublane
  tile_m = cfgs.tiles.tile_m
  tile_k = cfgs.tiles.tile_k

  def _inner(
      tiled_rhs_ref,
      tiled_out_ref,
      partial_out_ref,
      acc_ref,
      metadata_ref,
      lhs_u32_ref,   # VMEM [tile_m, tile_k] uint32 (over-fetch staging)
      lhs_bf16_ref,  # VMEM [tile_m, tile_k] bf16 (unpacked lhs tile)
      idx_smem_ref,  # SMEM [2*tile_m + 128] int32 (staged sidecar window)
      gsem_ref,      # DMA sem [1]
  ):
    gm_id = pl.program_id(1)
    k_id = pl.program_id(2)
    gsem = gsem_ref.at[0]
    _, m_start, _ = derive_tile(metadata_ref, gm_id, cfgs)
    q0 = (m_start // sublane) * sublane  # first expanded row of the tile
    # Stage the sidecar window [2*q0, 2*q0 + 2*tile_m) via a 128-aligned fetch.
    want = side_elem_base + 2 * q0
    start0 = want // 128 * 128
    phase = want - start0
    sz = idx_smem_ref.shape[0]  # static 2*tile_m + 128
    cp = pltpu.make_async_copy(
        side_ref.at[pl.ds(start0, sz)], idx_smem_ref, gsem
    )
    cp.start()
    cp.wait()

    def _abs_row(rr):
      q = idx_smem_ref[phase + 2 * rr]  # dedup offset of expanded row q0+rr
      q = jnp.minimum(jnp.maximum(q, 0), clamp_rows - 1)  # ALWAYS clamp
      return x_row_base + q

    def _gcp(rr):
      a = _abs_row(rr)
      return pltpu.make_async_copy(
          x_u32.at[pl.ds(a // 2, 1), pl.ds(k_id * tile_k, tile_k)],
          lhs_u32_ref.at[pl.ds(rr, 1)],
          gsem,
      )

    for rr in range(tile_m):
      _gcp(rr).start()
    for rr in range(tile_m):
      _gcp(rr).wait()
    # Unpack the wanted bf16 half per row (scalar shift — no [:, None]).
    for rr in range(tile_m):
      shift = (_abs_row(rr) & 1) * 16
      lhs_u32_ref[rr, :] = jnp.bitwise_and(lhs_u32_ref[rr, :] >> shift, 0xFFFF)
    lhs_bf16_ref[...] = jax.lax.bitcast_convert_type(
        lhs_u32_ref[...].astype(jnp.uint16), jnp.bfloat16
    )
    inner_kernel(
        lhs_bf16_ref,
        tiled_rhs_ref,
        tiled_out_ref,
        partial_out_ref,
        acc_ref,
        metadata_ref,
        cfgs=cfgs,
    )

  return _inner


def k1v1_kernel_main(
    # Scalar prefetch (1-D SMEM, arithmetic indexing only)
    c_flat_ref: jax.Array,  # int32[EP*EP*E_local] — expanded counts C (v0)
    d_flat_ref: jax.Array,  # int32[EP*EP] — dedup counts Ddup[src, dst]
    my_id_ref: jax.Array,   # int32[1]
    # In
    x_dedup_ref: jax.Array,   # [EP*cap_dedup, K] HBM — dedup rows, BLK-aligned
    # segments (prep emits this layout directly; nothing is repacked here)
    side_send_ref: jax.Array,  # int32[2*m_al + PAD] HBM — interleaved sidecar
    rev_send_ref: jax.Array,   # int32[topk*EP*cap_dedup + PAD] HBM — rev map
    rhs_ref: WeightsRef,       # weight [E_local, K, N] HBM
    # Out
    #   y_out_ref   [EP*cap_rows, aligned_n] — EXPANDED slot layout, == v0
    #   x_recv_ref  [EP*cap_dedup, K] — dedup recv slots (remote-DMA'd)
    #   side_recv_ref int32[EP*2*cap_rows + PAD] — sidecar recv slots
    #   rev_recv_ref  int32[EP*topk*cap_dedup + PAD] — rev recv slots
    y_out_ref: jax.Array,
    x_recv_ref: jax.Array,
    side_recv_ref: jax.Array,
    rev_recv_ref: jax.Array,
    # Scratch
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    zero_ref: jax.Array,
    zero_sem_ref: jax.Array,
    send_sem: jax.Array,
    recv_sem: jax.Array,
    lhs_u32_ref: jax.Array,
    lhs_bf16_ref: jax.Array,
    idx_smem_ref: jax.Array,
    gsem_ref: jax.Array,
    *,
    cfgs: GmmConfigs,  # PER-SEGMENT configs over the EXPANDED rows (size_m=CAP)
    ep: int,
    cap_rows: int,    # EXPANDED slot rows (v0 CAP)
    cap_dedup: int,   # dedup slot rows (align_up(T, BLK))
    topk: int,
    blk: int,
    device_id_fn=None,
    mode: str,        # "full" | "nosend" | "sendonly" | "nowait" (v0 semantics)
):
  """K1-v1 body: barrier -> 3-stream blocked sends -> EP gated gather-pipelines
  -> drain. Structure/order identical to k1_kernel_main; only the lhs source
  (gathered dedup rows) and the extra sidecar streams differ."""
  e_local = cfgs.dims.size_lhs_group
  size_k = cfgs.dims.size_k
  sublane = cfgs.dims.size_lhs_sublane
  num_k = pl.cdiv(size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

  my = my_id_ref[0]
  if device_id_fn is None:
    device_id_fn = lambda r: (r,)

  if cfgs.rhs_cfgs.should_bitcast:
    rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
    rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

  ch3 = blk // sublane
  x_dedup3 = x_dedup_ref.reshape(-1, sublane, size_k)
  x_recv3 = x_recv_ref.reshape(-1, sublane, size_k)
  xdd_u32 = x_dedup_ref.bitcast(jnp.uint32)   # [EP*cap_dedup//2, K]
  xrecv_u32 = x_recv_ref.bitcast(jnp.uint32)

  # ---- 1. Barrier (v0 idiom)
  if mode != "nosend":
    bsem = pltpu.get_barrier_semaphore()
    for d in range(ep):
      pl.semaphore_signal(
          bsem, inc=1, device_id=device_id_fn(d), device_id_type=DEVICE_MESH
      )
    pl.semaphore_wait(bsem, ep)

  # ---- counts arithmetic
  def _seg_rows(base):
    return lax.fori_loop(
        0, e_local, lambda e, a: a + c_flat_ref[base + e], jnp.int32(0)
    )

  # Per-dst block counts + aligned starts (rows), both streams' layouts.
  exp_nblk, exp_start = [], []   # expanded sidecar segments
  dd_nblk, dd_start = [], []     # dedup rows / rev segments
  es = jnp.int32(0)
  ds_ = jnp.int32(0)
  for dst in range(ep):
    nbe = pl.cdiv(_seg_rows((my * ep + dst) * e_local), blk)
    exp_nblk.append(nbe)
    exp_start.append(es)
    es = es + nbe * blk
    nbd = pl.cdiv(d_flat_ref[my * ep + dst], blk)
    dd_nblk.append(nbd)
    dd_start.append(ds_)
    ds_ = ds_ + nbd * blk

  def _sel(lst, idx):
    v = lst[0]
    for j in range(1, ep):
      v = jnp.where(idx == j, lst[j], v)
    return v

  my_dd_start = _sel(dd_start, my)     # rows
  my_exp_start = _sel(exp_start, my)   # rows

  # ---- 2. Sends: ring order (v0's proven arrival staircase), three streams
  # per dst on the SAME semaphores (byte accounting mirrors in the drains).
  if mode != "nosend":
    for d in range(1, ep):
      dst = jnp.mod(my + jnp.int32(d), jnp.int32(ep))
      did = device_id_fn(dst)
      nb_r = _sel(dd_nblk, dst)
      st_r = _sel(dd_start, dst)
      st_r3 = st_r // sublane

      def _rows_blocks(i, carry, dst=dst, st_r3=st_r3, did=did):
        pltpu.make_async_remote_copy(
            x_dedup3.at[pl.ds(st_r3 + i * ch3, ch3)],
            x_recv3.at[pl.ds(my * (cap_dedup // sublane) + i * ch3, ch3)],
            send_sem.at[dst],
            recv_sem.at[my],
            device_id=did,
            device_id_type=DEVICE_MESH,
        ).start()
        return carry

      lax.fori_loop(0, nb_r, _rows_blocks, jnp.int32(0))

      nb_e = _sel(exp_nblk, dst)
      st_e = _sel(exp_start, dst)
      blk2 = 2 * blk

      def _side_blocks(i, carry, dst=dst, st_e=st_e, did=did):
        pltpu.make_async_remote_copy(
            side_send_ref.at[pl.ds(st_e * 2 + i * blk2, blk2)],
            side_recv_ref.at[pl.ds(my * (2 * cap_rows) + i * blk2, blk2)],
            send_sem.at[dst],
            recv_sem.at[my],
            device_id=did,
            device_id_type=DEVICE_MESH,
        ).start()
        return carry

      lax.fori_loop(0, nb_e, _side_blocks, jnp.int32(0))

      blkk = topk * blk

      def _rev_blocks(i, carry, dst=dst, st_r=st_r, did=did):
        pltpu.make_async_remote_copy(
            rev_send_ref.at[pl.ds(st_r * topk + i * blkk, blkk)],
            rev_recv_ref.at[pl.ds(my * (topk * cap_dedup) + i * blkk, blkk)],
            send_sem.at[dst],
            recv_sem.at[my],
            device_id=did,
            device_id_type=DEVICE_MESH,
        ).start()
        return carry

      lax.fori_loop(0, nb_r, _rev_blocks, jnp.int32(0))

  # Drain-by-bytes: one reconstructed copy per stream (v0 idiom, 3x).
  def _drain(sem_view, nb_dd, nb_exp):
    r3 = nb_dd * ch3
    pltpu.make_async_copy(
        x_recv3.at[pl.ds(0, r3)], x_recv3.at[pl.ds(0, r3)], sem_view
    ).wait()
    e = nb_exp * 2 * blk
    pltpu.make_async_copy(
        side_recv_ref.at[pl.ds(0, e)], side_recv_ref.at[pl.ds(0, e)], sem_view
    ).wait()
    v = nb_dd * topk * blk
    pltpu.make_async_copy(
        rev_recv_ref.at[pl.ds(0, v)], rev_recv_ref.at[pl.ds(0, v)], sem_view
    ).wait()

  # ---- 3. Per-src compute: v0's consume order; lhs comes via the gather.
  out_in = y_out_ref.reshape(-1, sublane, y_out_ref.shape[-1])
  scratches = [
      partial_out_ref, acc_ref, metadata_ref,
      lhs_u32_ref, lhs_bf16_ref, idx_smem_ref, gsem_ref,
  ]

  for src_pos in range(ep):
    src = (my - src_pos + ep) % ep
    base_c = (src * ep + my) * e_local
    if mode not in ("nosend", "nowait") and src_pos != 0:
      _drain(
          recv_sem.at[src],
          pl.cdiv(d_flat_ref[src * ep + my], blk),
          pl.cdiv(_seg_rows(base_c), blk),
      )
    if mode == "sendonly":
      continue

    num_gm = fill_metadata(
        c_flat_ref, None, metadata_ref, cfgs=cfgs, base=base_c
    )
    row_base = src * (cap_rows // sublane)
    if cfgs.zero_init:
      zero_size = zero_out_start(
          y_out_ref, zero_ref, zero_sem_ref, metadata_ref, num_gm, cfgs=cfgs,
          row_offset=src * cap_rows, num_rows=cap_rows,
      )

    if src_pos == 0:  # STATICALLY the self segment — read send-side buffers
      x_u32, x_row_base = xdd_u32, my_dd_start
      side_ref, side_base = side_send_ref, my_exp_start * 2
    else:
      x_u32, x_row_base = xrecv_u32, src * cap_dedup
      side_ref, side_base = side_recv_ref, src * (2 * cap_rows)

    inner = make_v1_gather_inner(
        cfgs, x_u32, side_ref, x_row_base, side_base, cap_dedup
    )
    (_, rhs_spec), out_spec = generate_block_specs(
        metadata_ref, cfgs, row_base=row_base
    )
    pipeline_fn = pltpu.emit_pipeline(
        inner,
        grid=(num_n, num_gm, num_k),
        in_specs=(rhs_spec,),
        out_specs=out_spec,
    )
    pipeline_fn(rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
      zero_out_end(y_out_ref, zero_sem_ref, zero_size, dims=cfgs.dims)

  # ---- 4. Drain sends (self slot holds 0 signals — zero-size waits).
  if mode != "nosend":
    for dst in range(ep):
      z = jnp.int32(0)
      nb_dd = jnp.where(my == dst, z, dd_nblk[dst])
      nb_exp = jnp.where(my == dst, z, exp_nblk[dst])
      _drain(send_sem.at[dst], nb_dd, nb_exp)

  # ---- 5. "nowait": drain the never-waited recv sems before retiring.
  if mode == "nowait":
    for src_pos in range(1, ep):
      src = (my + src_pos) % ep
      _drain(
          recv_sem.at[src],
          pl.cdiv(d_flat_ref[src * ep + my], blk),
          pl.cdiv(_seg_rows((src * ep + my) * e_local), blk),
      )


V1_SIDE_PAD = 128  # int32 slack (mirrors prep_a2a.V1_SIDE_PAD; keep in sync)


def gmm_a2a_v1(
    x_dedup: jax.Array,   # [EP*cap_dedup, K] bf16 — dedup rows (prep layout)
    counts_c: jax.Array,  # int32[EP, EP, E_local] — gathered expanded counts
    dedup_d: jax.Array,   # int32[EP, EP] — gathered dedup counts Ddup[src, dst]
    side_int: jax.Array,  # int32[2*m_al + PAD] — interleaved (idx, w) sidecar
    rev_int: jax.Array,   # int32[topk*EP*cap_dedup + PAD] — reverse sidecar
    rhs: jax.Array,       # [E_local, K, N] bf16
    *,
    my_id: jax.Array,
    ep: int,
    cap_rows: int,    # EXPANDED slot rows; must be %BLK (pass align_up(M, BLK))
    cap_dedup: int,   # dedup slot rows = align_up(T, BLK) (prep's value)
    topk: int,
    collective_id: int = 46,
    blk: int = 512,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    mode: str = "full",
    return_recv: bool = False,  # also return x_recv (dedup recv slots; slot
    # [my] NEVER written — self-segment direct read; bwd must patch it)
    ep_axis_name: str | None = None,
    mesh_axis_names: tuple | None = None,
    mesh_shape=None,
    interpret: bool = False,
):
  """Fused v1 dedup dispatch: in-kernel 3-stream a2a + gather-lhs gmm.

  Returns (y_out [EP, cap_rows, N], side_recv, rev_recv) — the recv sidecars
  feed gmm_return_v1 (rev) and the w_wide epilogue scale / backward (side).
  With return_recv=True: (y_out, side_recv, rev_recv, x_recv).
  Bit-exactness target: v0 gmm_a2a's y_out on the same routing (gate 2).
  """
  assert mode in ("full", "nosend", "sendonly", "nowait"), mode
  size_r, size_k = x_dedup.shape
  e_local, size_k2, size_n = rhs.shape
  assert size_k2 == size_k
  assert size_r == ep * cap_dedup, (x_dedup.shape, ep, cap_dedup)
  assert counts_c.shape == (ep, ep, e_local), counts_c.shape
  assert counts_c.dtype == jnp.int32
  assert dedup_d.shape == (ep, ep) and dedup_d.dtype == jnp.int32
  assert rev_int.shape == (topk * ep * cap_dedup + V1_SIDE_PAD,), rev_int.shape
  assert side_int.dtype == jnp.int32 and rev_int.dtype == jnp.int32
  m_al = (side_int.shape[0] - V1_SIDE_PAD) // 2
  assert side_int.shape[0] == 2 * m_al + V1_SIDE_PAD, side_int.shape
  assert m_al % blk == 0, (m_al, blk)

  if tile_info is calculate_tiling and size_k == 7168:
    tile_info = TileSizes(tile_m=256, tile_k=7168, tile_n=1024)

  assert cap_rows % blk == 0 and cap_dedup % blk == 0, (cap_rows, cap_dedup)
  itemsize = jnp.dtype(x_dedup.dtype).itemsize
  assert (blk * size_k * itemsize) % 128 == 0
  # 1-D int32 stream granularity: every element offset/size must be %128.
  assert (2 * blk) % 128 == 0, blk        # sidecar blocks (blk >= 64)
  assert (topk * blk) % 128 == 0, (topk, blk)  # rev blocks
  assert (2 * cap_rows) % 128 == 0 and (topk * cap_dedup) % 128 == 0

  cfgs, vmem_limit_bytes = k1_make_cfgs(
      x_dedup.dtype, rhs.dtype, size_k, size_n, e_local, cap_rows,
      blk=blk, tile_info=tile_info, vmem_limit_bytes=vmem_limit_bytes,
  )
  dims = cfgs.dims
  tiles = cfgs.tiles
  assert cap_rows % dims.size_lhs_sublane == 0
  assert blk % dims.size_lhs_sublane == 0
  assert cap_dedup % 2 == 0  # u32 pair-packing
  assert (2 * tiles.tile_m) % 128 == 0, tiles  # SMEM staging window
  assert tiles.tile_k % 128 == 0, tiles        # u32 gather column slice

  c_flat = counts_c.reshape(-1)
  d_flat = dedup_d.reshape(-1)
  my_id_arr = jnp.asarray(my_id, jnp.int32).reshape(1)

  num_lanes = pltpu.get_tpu_info().num_lanes
  target_zero_ref_bytes = 32 * 1024
  out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
  tile_zero_m = min(target_zero_ref_bytes // num_lanes // out_bytes, dims.size_m)

  scratch_shapes = [
      pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
      pltpu.VMEM((tiles.tile_m, tiles.tile_n), cfgs.acc_dtype),
      MetadataRef(
          group_start=pltpu.SMEM((dims.size_lhs_group + 1,), jnp.int32),
          gm_prefix=pltpu.SMEM((dims.size_lhs_group + 1,), jnp.int32),
          group_offset=pltpu.SMEM((1,), jnp.int32),
          bounds=pltpu.SMEM((2,), jnp.int32),
      ),
      pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
      pltpu.SemaphoreType.DMA((1,)),
      pltpu.SemaphoreType.DMA((ep,)),
      pltpu.SemaphoreType.DMA((ep,)),
      # V1 gather scratch
      pltpu.VMEM((tiles.tile_m, tiles.tile_k), jnp.uint32),
      pltpu.VMEM((tiles.tile_m, tiles.tile_k), jnp.bfloat16),
      pltpu.SMEM((2 * tiles.tile_m + 128,), jnp.int32),
      pltpu.SemaphoreType.DMA((1,)),
  ]

  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_init = (
      jax.ShapeDtypeStruct((ep * cap_rows, aligned_n), cfgs.out_dtype),
      jax.ShapeDtypeStruct((ep * cap_dedup, size_k), x_dedup.dtype),
      jax.ShapeDtypeStruct((ep * 2 * cap_rows + V1_SIDE_PAD,), jnp.int32),
      jax.ShapeDtypeStruct((ep * topk * cap_dedup + V1_SIDE_PAD,), jnp.int32),
  )
  rhs_weights = WeightsRef(weight=rhs, scale=None, bias=None)

  y_out, x_recv, side_recv, rev_recv = pl.pallas_call(
      functools.partial(
          k1v1_kernel_main, cfgs=cfgs, ep=ep, cap_rows=cap_rows,
          cap_dedup=cap_dedup, topk=topk, blk=blk,
          device_id_fn=make_device_id_fn(
              ep_axis_name, mesh_axis_names, mesh_shape
          ),
          mode=mode,
      ),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
              WeightsRef(
                  weight=pl.BlockSpec(memory_space=pltpu.HBM),
                  scale=None,
                  bias=None,
              ),
          ],
          out_specs=tuple(
              pl.BlockSpec(memory_space=pltpu.HBM) for _ in out_init
          ),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          collective_id=None if mode == "nosend" else collective_id,
      ),
      name=f"k1v1_a2a_ep{ep}_{mode}_blk{blk}-{get_scope_name(cfgs)}",
      cost_estimate=_k1_cost_estimate(cfgs, ep),
      metadata=get_metadata(cfgs),
      interpret=interpret,
  )(c_flat, d_flat, my_id_arr, x_dedup, side_int, rev_int, rhs_weights)

  y = y_out[:, : cfgs.out_size_n].reshape(ep, cap_rows, cfgs.out_size_n)
  if return_recv:
    return y, side_recv, rev_recv, x_recv.reshape(ep, cap_dedup, size_k)
  return y, side_recv, rev_recv
