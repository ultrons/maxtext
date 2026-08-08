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
# k2_fused.py — Kernel 2: fused return (wo-gmm + a2a-back), K2_KERNEL_SPEC.md /
# WIRE_FORMAT.md §3+§6.
#
# Mirror of k1_fused.gmm_a2a. ALL fork machinery is imported from k1_fused
# (single-sourced): MetadataRef, fill_metadata-with-base, derive_tile (via
# generate_block_specs/inner_kernel), zero_out_start/end, k1_make_cfgs, the
# blocked remote-DMA + drain-by-bytes idioms, the mode plumbing. k1_fused.py is
# NOT modified.
#
# STRUCTURE: one grid-less pallas_call whose body sequences
#   barrier -> EP x (per-src emit_pipeline (wo-gmm) -> blocked remote SEND of
#   that segment's output back to src) -> send drain -> recv drain.
# Segment i's return flight hides under segment i+1's gmm (K1's pipeline
# mirrored). The LAST segment's send + the peers' inbound tail are the accepted
# v0 exposed tail (no intra-segment column-stripe streaming in v0 — the
# (n, gm, k) grid completes rows only at the last n step; stripe sends are a
# v1 idea, noted, not built).
#
# Buffers (mirror of K1's deviations, same reasons):
#   * y_local input is handled internally as a FLAT [EP*CAP, K2] buffer; the
#     per-src base offset is folded into the (forked) index maps.
#   * y_stage [EP*CAP, aligned_n]: per-src staging slots for the return sends,
#     declared as a (discarded) pallas OUTPUT (Mosaic cannot allocate HBM/ANY
#     scratch — K1's x_recv pattern). Slot src holds segment src's wo-gmm
#     output rows [0, C[src,my].sum()), BLK-aligned sends read [0, align_up).
#   * y_home [M + EP*BLK, aligned_n]: the real output — MY rows back in MY
#     send order, in K1's x_send BLK-ALIGNED segment layout: segment d (rows I
#     sent to dst d) starts at sum_{d'<d} align_up(C[my,d'].sum(), BLK). It is
#     written by REMOTE DMAs from every peer (at MY aligned offset in each) and
#     LOCALLY by the self-segment pipeline (the src==my mirror of K1's
#     self-segment fix: the own segment never travels).
#   * All a2a DMAs run on 3-D sublane views; every offset is a BLK multiple
#     (BLK % sublane == 0), so Mosaic's tiled-HBM slice constraint is satisfied
#     by construction — no XLA-level repack is needed on the return path
#     (y_home is DEFINED aligned).
#
# DESTINATION-OFFSET ARITHMETIC (the silent-corruption trap): the return send
# for segment src lands at src's y_home offset
#     ret_off(src, my) = sum_{d' < my} align_up(C[src, d'].sum(), BLK)
# — src's send-order segment for dst == my, the SAME C table and the SAME
# align_up convention as K1's x_send repack (al_starts = cumsum(align_up(
# send_sizes, BLK))). Checked against prep_a2a's ragged segment math
# (my_starts = cumsum(send_sizes) - send_sizes) by the combine's ragged->
# aligned position map in check_k2.py's round trip.
#
# CONSUME ORDER (kept from K1's proven discipline): own segment first, then
# src = (my - d) % EP at stage d. My return send to src fires at stage
# d = (my - src) % EP, i.e. receiver r gets its inbound segments staircased
# (from r+1 early ... r+ (EP-1) late) — send timing and the receivers' final
# drain tail stay compatible by the same symmetry that fixed K1's 5.1ms.
#
# MODES (static python values, each its own trace; K1 semantics transposed):
#   "full"     barrier -> EP x (gmm -> send) -> drain sends -> drain recvs.
#              The only checked-for-correctness mode.
#   "nosend"   no barrier/sends/drains; the EP wo-gmm pipelines only (self
#              segment still lands in y_home; remote segments in y_stage).
#              Output INCOMPLETE (perf instrument: compute floor + overheads).
#   "sendonly" barrier + sends of the (garbage, unwritten) staging slots +
#              drains, no pipelines. Output GARBAGE (pure transport).
#   "nowait"   OVERLAP PROBE, K1's semantics transposed: ALL return sends fire
#              upfront (of garbage/racy staging — sends never wait for their
#              gmm), pipelines run back-to-back, send+recv sems drained only at
#              the very end. Output GARBAGE. Measures max(compute, transport)
#              if the engines overlap, compute + transport if they serialize.
#              (K2 "full" is already fire-and-forget with drain-at-end — the
#              distinct probe is un-gating the sends from their own segment's
#              compute.)
#
# Repeated-invocation semaphore hygiene: every mode that sends drains BOTH
# send_sem (by my outbound byte counts) and recv_sem (by every peer's inbound
# byte counts, both sides derived from the same C) before kernel retire.
import dataclasses
import functools
from typing import Tuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .k1_fused import (  # single-sourced fork machinery — see header
    DEVICE_MESH,
    GmmConfigs,
    MetadataRef,
    TileFn,
    TileSizes,
    V1_SIDE_PAD,
    WeightsRef,
    align_to,
    calculate_tiling,
    fill_metadata,
    generate_block_specs,
    get_metadata,
    get_scope_name,
    inner_kernel,
    inner_kernel_rs,
    k1_make_cfgs,
    make_device_id_fn,
    make_lhs_scale_spec,
    zero_out_end,
    zero_out_start,
)


def k2_kernel_main(
    # Scalar prefetch (1-D SMEM, arithmetic indexing only)
    c_flat_ref: jax.Array,  # int32[EP*EP*E_local] — gathered counts C (§2),
    # C[src, dst, e] at index (src*EP + dst)*E_local + e (SAME table K1 used)
    my_id_ref: jax.Array,  # int32[1] — this device's EP rank
    # In
    y_local_ref: jax.Array,  # [EP*CAP, K2] HBM — K1's output slot layout,
    # flat; segment src occupies rows [src*CAP, (src+1)*CAP), fill =
    # C[src, my].sum() expert-sorted rows (§3)
    rhs_ref: WeightsRef,  # weight [E_local, K2, N2] HBM — local wo weights
    # Out
    y_home_ref: jax.Array,  # [M + EP*BLK, aligned_n] HBM — MY rows back in MY
    # send order, BLK-ALIGNED segments (K1's x_send layout). Written by REMOTE
    # DMAs from peers + locally by the self-segment pipeline.
    y_stage_ref: jax.Array,  # [EP*CAP, aligned_n] HBM — per-src staging slots
    # for the return sends; a (discarded) pallas OUTPUT (K1's x_recv pattern:
    # Mosaic cannot allocate HBM/ANY scratch here).
    # Scratch
    partial_out_ref: jax.Array,  # [size_lhs_sublane, tile_n] VMEM (shared, 4x)
    acc_ref: jax.Array,  # [tile_m, tile_n] VMEM (shared, 4x)
    metadata_ref: MetadataRef,  # SMEM, refilled per src segment
    zero_ref: jax.Array,  # [tile_zero_m, num_lanes] VMEM
    zero_sem_ref: jax.Array,  # DMA sem [1] for the zero-fill copies
    send_sem: jax.Array,  # DMA sem [EP], slot dst — signaled as my sends land
    recv_sem: jax.Array,  # DMA sem [EP], slot = SENDER's rank
    *,
    cfgs: GmmConfigs,  # PER-SEGMENT configs: dims.size_m == CAP,
    # dims.size_group == dims.size_lhs_group == E_local
    ep: int,
    cap_rows: int,
    blk: int,
    device_id_fn=None,  # rank -> DeviceIdType.MESH tuple (make_device_id_fn);
    # None => single-axis "ep" (rank,). Multi-axis meshes MUST pass a real one.
    mode: str,  # "full" | "nosend" | "sendonly" | "nowait" | "norecvdrain" — STATIC
    self_last: bool = False,  # OVERLAP FIX: compute the self segment (no send)
    # LAST instead of first, so every remote send has a following gmm to hide
    # under AND (symmetric) every device's sends fire ~1 segment earlier ->
    # peers' inbound arrives earlier too. Bit-exact (same math, async sends
    # drained at end; distinct out targets are order-independent).
):
  """K2 body: barrier -> EP x (wo-gmm pipeline -> return send) -> drains."""
  e_local = cfgs.dims.size_lhs_group
  size_k = cfgs.dims.size_k
  sublane = cfgs.dims.size_lhs_sublane
  num_k = pl.cdiv(size_k, cfgs.tiles.tile_k)
  num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

  my = my_id_ref[0]
  if device_id_fn is None:
    device_id_fn = lambda r: (r,)  # single-axis "ep" default (check_k2)

  # Keep the fp8/packed plumbing intact (no-op for bf16), as in kernel_main.
  if cfgs.rhs_cfgs.should_bitcast:
    rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
    rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

  # 3-D SUBLANE VIEWS for all a2a DMAs (K1's zero_out idiom): the leading dim
  # carries no tile-alignment constraint; every offset below is a multiple of
  # ch3 (BLK-aligned layout by construction, BLK % sublane == 0).
  ch3 = blk // sublane  # DMA block size in tile-row units (static)
  width = y_home_ref.shape[-1]  # aligned_n (== y_stage width)
  y_home3 = y_home_ref.reshape(-1, sublane, width)
  y_stage3 = y_stage_ref.reshape(-1, sublane, width)
  lhs_in = y_local_ref.reshape(-1, sublane, size_k)
  cap3 = cap_rows // sublane

  # ---- 1. Barrier (once): no peer may still be reading its y_home from the
  # previous invocation when our return sends start overwriting it. (Skipped in
  # "nosend" — no remote traffic exists to fence.)
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

  def _ret_off3(src):
    """MY aligned segment start inside src's y_home, tile-row units.

    = sum_{d' < my} align_up(C[src, d'].sum(), BLK) — src's send-order segment
    for dst == my, SAME C table + align_up convention as K1's x_send repack
    (al_starts = cumsum(align_up(send_sizes, BLK)); see header trap note).
    `src` and `my` are traced; d' is the static loop.
    """
    off = jnp.int32(0)
    for dp in range(ep):
      nblk = pl.cdiv(_seg_rows((src * ep + jnp.int32(dp)) * e_local), blk)
      off = off + jnp.where(jnp.int32(dp) < my, nblk * ch3, jnp.int32(0))
    return off

  def _send_segment(src, base_c):
    """Blocked return send: segment src's staged output -> src's y_home.

    Semaphore discipline UNCHANGED from K1: send_sem slot = dst (== src of the
    segment), recv_sem slot = SENDER's rank (my, on the receiver's array).
    src != my structurally at every call site (the self segment never travels).
    """
    nblk = pl.cdiv(_seg_rows(base_c), blk)  # traced; tail block carries slack
    dst_off3 = _ret_off3(src)
    src_slot3 = src * cap3

    def _send_blocks(i, carry, src=src, dst_off3=dst_off3,
                     src_slot3=src_slot3):
      pltpu.make_async_remote_copy(
          y_stage3.at[pl.ds(src_slot3 + i * ch3, ch3)],
          y_home3.at[pl.ds(dst_off3 + i * ch3, ch3)],
          send_sem.at[src],
          recv_sem.at[my],
          device_id=device_id_fn(src),
          device_id_type=DEVICE_MESH,
      ).start()
      return carry

    lax.fori_loop(0, nblk, _send_blocks, jnp.int32(0))

  # Drain-by-bytes for DMA semaphores (K1's reconstructed-copy idiom; jax
  # 0.10.1 rejects pl.semaphore_wait on DMA sems). Block bytes here are
  # BLK * aligned_n * itemsize — sized off the y_stage3 view (same width as
  # y_home3, so send and recv drains reconstruct identical byte counts).
  def _drain_dma_sem(sem_view, nblocks):
    rows3 = nblocks * ch3
    pltpu.make_async_copy(
        y_stage3.at[pl.ds(0, rows3)],  # refs only size the wait
        y_stage3.at[pl.ds(0, rows3)],
        sem_view,
    ).wait()

  # Self segment's aligned start in MY OWN y_home (src == my in _ret_off3):
  # the src_pos == 0 pipeline writes straight there — mirror of K1's self fix.
  my_ret_start3 = _ret_off3(my)

  # ---- "nowait" OVERLAP PROBE: ALL return sends fire upfront, un-gated from
  # their segment's compute (staging contents garbage/racy — see header). Same
  # blocks/sems/offsets as full, same issue order (my-1, my-2, ...).
  if mode == "nowait":
    for src_pos in range(1, ep):
      src = (my - src_pos + ep) % ep  # traced
      _send_segment(src, (src * ep + my) * e_local)

  # ---- 2. Per-src compute + interleaved return sends: static unroll. Default
  # own-segment-first (K1's proven order); self_last moves it to the end so the
  # last remote send hides under the self gmm (OVERLAP FIX).
  scratches = [partial_out_ref, acc_ref, metadata_ref]

  order = [*range(1, ep), 0] if self_last else list(range(ep))
  for src_pos in order:
    src = (my - src_pos + ep) % ep  # traced
    base_c = (src * ep + my) * e_local  # group sizes C[src, my, :]

    if mode != "sendonly":  # transport only: no metadata/zero/pipeline
      num_gm = fill_metadata(
          c_flat_ref, None, metadata_ref, cfgs=cfgs, base=base_c
      )

      # Output target: self segment writes DIRECTLY into y_home at my aligned
      # segment start (BLK-aligned => sublane-aligned; traced row_base is the
      # K1-proven path); remote segments write their y_stage slot.
      if src_pos == 0:  # STATICALLY the self segment (src == my)
        out_ref2, out3 = y_home_ref, y_home3
        row_base = my_ret_start3
        zero_row_offset = my_ret_start3 * sublane
        # zero-fill bound = the aligned segment span (slack rows are sent by
        # tail blocks on the remote side / never read by the combine).
        zero_num_rows = pl.cdiv(_seg_rows(base_c), blk) * blk
      else:
        out_ref2, out3 = y_stage_ref, y_stage3
        row_base = src * cap3
        zero_row_offset = src * cap_rows
        zero_num_rows = cap_rows

      if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref2,
            zero_ref,
            zero_sem_ref,
            metadata_ref,
            num_gm,
            cfgs=cfgs,
            row_offset=zero_row_offset,
            num_rows=zero_num_rows,
        )

      # lhs always reads y_local slot src (the wo-gmm consumes K1's output in
      # place); the out row base differs (y_home vs y_stage) — the transposed
      # use of K1's row_base / lhs_row_base split.
      (lhs_spec, rhs_spec), out_spec = generate_block_specs(
          metadata_ref, cfgs, row_base=row_base, lhs_row_base=src * cap3
      )
      pipeline_fn = pltpu.emit_pipeline(
          functools.partial(inner_kernel, cfgs=cfgs),
          grid=(num_n, num_gm, num_k),
          in_specs=(lhs_spec, rhs_spec),
          out_specs=out_spec,
      )
      # emit_pipeline's finalize step waits the last output copy before
      # returning (verified in jax 0.10.1 pipeline.py::finalize), so the
      # staged rows are in HBM when the send below reads them.
      pipeline_fn(lhs_in, rhs_ref, out3, scratches=scratches)

      if cfgs.zero_init:
        zero_out_end(out_ref2, zero_sem_ref, zero_size, dims=cfgs.dims)

    # Return send for THIS segment (full: right after its pipeline completes —
    # segment i's flight hides under segment i+1's gmm; sendonly: same loop
    # position, garbage staging). Self segment never travels. "nowait" already
    # sent everything upfront.
    if mode in ("full", "sendonly", "norecvdrain") and src_pos != 0:
      _send_segment(src, base_c)

  # ---- 3. Drains: never retire with in-flight DMAs / non-zero sems.
  if mode != "nosend":
    # Sends: dst d got cdiv(C[d, my].sum(), BLK) blocks from me (its rows I
    # computed); self slot holds 0 signals (never sent) — traced-size 0 wait.
    for d in range(ep):
      nblk_send = jnp.where(
          my == jnp.int32(d),
          jnp.int32(0),
          pl.cdiv(_seg_rows((jnp.int32(d) * ep + my) * e_local), blk),
      )
      _drain_dma_sem(send_sem.at[d], nblk_send)
    # Recvs: peer p returns MY rows I sent it = cdiv(C[my, p].sum(), BLK)
    # blocks into my y_home. This wait is what makes y_home complete at kernel
    # retire (the accepted v0 exposed tail lives here). "norecvdrain" = PERF
    # PROBE: skip ONLY this loop (keep sends + send-drain) -> output GARBAGE
    # (peers still write y_home after retire). full - norecvdrain = the inbound
    # tail cost; norecvdrain - nosend = the outbound exposure (advisor split).
    if mode != "norecvdrain":
      for p in range(ep):
        nblk_recv = jnp.where(
            my == jnp.int32(p),
            jnp.int32(0),
            pl.cdiv(_seg_rows((my * ep + jnp.int32(p)) * e_local), blk),
        )
        _drain_dma_sem(recv_sem.at[p], nblk_recv)


def _k2_cost_estimate(cfgs: GmmConfigs, ep: int) -> pl.CostEstimate:
  """Upper-bound cost: EP segments of up to CAP rows each + the a2a bytes."""
  dims = cfgs.dims
  flops = 2 * ep * dims.size_m * dims.size_k * dims.size_n
  lhs_bytes = (
      ep * dims.size_m * dims.size_k * jnp.dtype(cfgs.lhs_cfgs.dtype).itemsize
  )
  rhs_bytes = dims.size_group * dims.size_k * dims.size_n * jnp.dtype(
      cfgs.rhs_cfgs.dtype
  ).itemsize
  out_bytes = (
      ep * dims.size_m * cfgs.out_size_n * jnp.dtype(cfgs.out_dtype).itemsize
  )
  return pl.CostEstimate(
      flops=flops,
      bytes_accessed=lhs_bytes + rhs_bytes + 2 * out_bytes,  # gmm + a2a-back
      transcendentals=0,
  )


def gmm_return(
    y_local: jax.Array,  # [EP, CAP, K2] bf16 — K1's output slot layout (§3)
    counts_c: jax.Array,  # int32[EP, EP, E_local] — the SAME C table K1 used
    rhs: jax.Array,  # [E_local, K2, N2] bf16 — local wo weights
    *,
    my_id: jax.Array,  # int32 scalar — this device's EP rank (lax.axis_index)
    ep: int,
    cap_rows: int,  # per-src slot rows; v0: CAP = M (dropless worst case)
    m_rows: int | None = None,  # MY total send rows (= T*k). None -> cap_rows
    # (v0 CAP = M). Static: sizes y_home = [m_rows + EP*BLK, N2].
    collective_id: int = 43,  # distinct from K1's 42 — both barriers coexist
    blk: int = 512,  # rows per DMA block (static slice size; starts are traced)
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    mode: str = "full",  # "full" | "nosend" | "sendonly" | "nowait" — STATIC
    # (see k2_kernel_main header; only "full" is checked for correctness)
    rhs_scale: jax.Array | None = None,  # PHASE E: [E_local, num_blocks, 1, N2]
    # f32 — fp8 wo-weight per-K-block scales (quant_utils.quantize_weights).
    # K2's lhs never travels (it is K1's LOCAL output), so there is NO sidecar
    # here: with fp8 rhs + maybe_quantize_lhs the fork's own in-kernel lhs
    # quantization path runs (per-512-block amax -> block_scale_inv), exactly
    # the machinery the fp8-gmm bench validated. Epilogue unchanged; the
    # return sends stay bf16 (v0).
    maybe_quantize_lhs: bool = True,  # PHASE E: the fork's flag, passed
    # through. Inert without rhs_scale (bf16 path unchanged).
    ep_axis_name: str | None = None,  # MULTI-AXIS mesh device addressing —
    mesh_axis_names: tuple | None = None,  # see gmm_a2a; all None => single-axis
    mesh_shape=None,  # "ep" (rank,). MaxText's full mesh MUST set these.
    self_last: bool = False,  # OVERLAP FIX (see k2_kernel_main): self segment last.
    interpret: bool = False,
) -> jax.Array:  # [m_rows + EP*BLK, N2] — MY rows back in MY send order,
  # BLK-ALIGNED segments (K1's x_send layout): segment d starts at
  # sum_{d'<d} align_up(C[my,d'].sum(), BLK). Slack rows are garbage/zero —
  # combine_home() maps ragged send positions onto this layout.
  """Fused return: chunk-by-src wo-gmm + in-kernel a2a-back (WIRE_FORMAT §6).

  Call under shard_map over the `ep` axis (check_rep/check_vma False), after
  gmm_a2a. Bit-exactness target: prep_a2a.reference_layer's return dataflow
  with THIS fork's gmm_v2 per segment for both matmuls (see check_k2.py).
  """
  assert mode in ("full", "nosend", "sendonly", "nowait", "norecvdrain"), mode
  assert y_local.ndim == 3 and y_local.shape[0] == ep, y_local.shape
  assert y_local.shape[1] == cap_rows, (y_local.shape, cap_rows)
  size_k = y_local.shape[2]
  e_local, size_k2, size_n = rhs.shape
  assert size_k2 == size_k, (rhs.shape, size_k)
  assert counts_c.shape == (ep, ep, e_local), counts_c.shape
  assert counts_c.dtype == jnp.int32

  if m_rows is None:
    m_rows = cap_rows  # v0: CAP = M

  # Production-tile default for the wo shape (K2=2048) = maxtext wo_tile_fwd
  # mapping (batch_seq=512 -> tm, mlp=2048 -> tk, embed=3584 -> tn). The earlier
  # tn=1024 guess made 7 n-tiles over the 7168-wide output -> 7x lhs re-reads
  # (MEASURED 13.8 ms/segment vs K1-side 4.9 at equal FLOPs); tn=3584 -> 2.
  # An EXPLICIT tile_info always wins; every other K keeps calculate_tiling.
  if tile_info is calculate_tiling and size_k == 2048:
    tile_info = TileSizes(tile_m=256, tile_k=2048, tile_n=3584)  # tm=512 scoped-VMEM-OOMs at prod (60MB>57.6)

  # Blocked-DMA layout guarantees: tail blocks carry up to BLK-1 slack rows;
  # y_home segments and y_stage slots are BLK-aligned by construction.
  assert m_rows % blk == 0, (m_rows, blk)
  assert cap_rows % blk == 0, (cap_rows, blk)
  assert cap_rows >= m_rows, (cap_rows, m_rows)  # v0: dropless worst case

  if rhs_scale is not None:
    assert rhs_scale.shape[0] == e_local and rhs_scale.shape[2:] == (1, size_n)

  # Per-SEGMENT gmm configs (dims.size_m = CAP) — k1_make_cfgs reused as-is;
  # the reference gmm_v2 in check_k2.py must be handed cfgs.tiles.
  cfgs, vmem_limit_bytes = k1_make_cfgs(
      y_local.dtype,
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
      maybe_quantize_lhs=maybe_quantize_lhs,
  )
  if rhs_scale is not None:
    # quant_block < mxu_column_size = the measured silent-bf16 cliff. Refuse.
    assert not cfgs.rhs_cfgs.should_dequantize_before_matmul, (
        "rhs quant_block", cfgs.rhs_cfgs.quant_block_size,
        "< mxu_column_size — the matmul would silently run bf16",
    )
  dims = cfgs.dims
  tiles = cfgs.tiles
  assert cap_rows % dims.size_lhs_sublane == 0
  assert blk % dims.size_lhs_sublane == 0

  num_lanes = pltpu.get_tpu_info().num_lanes
  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_itemsize = jnp.dtype(cfgs.out_dtype).itemsize
  assert (aligned_n * out_itemsize) % 128 == 0  # wire-row granule
  assert (blk * aligned_n * out_itemsize) % 128 == 0  # DMA granule

  c_flat = counts_c.reshape(-1)
  my_id_arr = jnp.asarray(my_id, jnp.int32).reshape(1)
  y_local_flat = y_local.reshape(ep * cap_rows, size_k)  # free (contiguous)

  target_zero_ref_bytes = 32 * 1024
  tile_zero_m = target_zero_ref_bytes // num_lanes // out_itemsize
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

  m_al = m_rows + ep * blk  # static bound: sum(align_up(rows_d, BLK)) <= this
  out_init = (
      jax.ShapeDtypeStruct((m_al, aligned_n), cfgs.out_dtype),
      # y_stage: return-send staging as a discarded output (see kernel header)
      jax.ShapeDtypeStruct((ep * cap_rows, aligned_n), cfgs.out_dtype),
  )
  rhs_weights = WeightsRef(
      weight=rhs,
      scale=None if rhs_scale is None else rhs_scale.astype(jnp.float32),
      bias=None,
  )

  y_home, _y_stage_discard = pl.pallas_call(
      functools.partial(
          k2_kernel_main, cfgs=cfgs, ep=ep, cap_rows=cap_rows, blk=blk,
          device_id_fn=make_device_id_fn(ep_axis_name, mesh_axis_names, mesh_shape),
          mode=mode, self_last=self_last,
      ),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),
              WeightsRef(
                  weight=pl.BlockSpec(memory_space=pltpu.HBM),
                  scale=(
                      None if rhs_scale is None
                      else pl.BlockSpec(memory_space=pltpu.HBM)
                  ),
                  bias=None,
              ),
          ],
          out_specs=(
              pl.BlockSpec(memory_space=pltpu.HBM),
              pl.BlockSpec(memory_space=pltpu.HBM),
          ),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
          # "nosend" has no barrier (get_barrier_semaphore never called);
          # Mosaic rejects a collective_id without a custom barrier.
          collective_id=None if mode == "nosend" else collective_id,
      ),
      name=f"k2_ret_ep{ep}_{mode}_blk{blk}-{get_scope_name(cfgs)}",
      cost_estimate=_k2_cost_estimate(cfgs, ep),
      metadata=get_metadata(cfgs),
      interpret=interpret,
  )(c_flat, my_id_arr, y_local_flat, rhs_weights)

  return y_home[:, : cfgs.out_size_n]


# =============================================================================
# WIRE-FORMAT V1 (WIRE_FORMAT_V1_SPEC.md §B): weighted bf16 partial return.
#
# Per src segment: wo-gmm (weights applied per EXPANDED row in the f32
# epilogue via the lhs_row_scale machinery — w_wide is the lane-replicated
# per-row weight buffer, built at XLA from K1-v1's recv'd sidecar) -> per-
# dedup-slot GATHER-REDUCE (the index-sidecar machinery of §A.3 in reverse:
# for each dedup slot, gather its <=topk already-weighted y rows by the
# slot-major rev sidecar and sum them in bf16, ascending local expert id;
# pad entries point at the segment's first zero_init'd slack row => exact
# zero terms; NO TC scatter anywhere) -> blocked return send of the DEDUP'd
# partial rows (Ddup counts).
#
# ARITHMETIC CONTRACT (the reference in check_k2.py implements exactly this;
# documented deviation from the spec's parenthetical "computed in bf16"
# multiply): partial[s] = fold_left_{j=0..topk-1} bf16-add of
#     y_scaled[rev[s*topk+j]],   y_scaled[p] = bf16(f32_gmm_acc[p] * w[p])
# i.e. the weight multiplies the f32 accumulator INSIDE the gmm epilogue
# (strictly more accurate than bf16(w)*bf16(y), reuses the proven fp8
# row-scale path); only the <=topk-term reduction and the <=EP-term home
# combine are bf16 — the user's bf16-reduce constraint (no f32 accumulation
# of wire data) holds.
#
# Home combine v1: out[t] = fold_left over PRESENT dsts (ascending src d) of
# bf16-add of partial_d[pos[t,d]] — absent dsts are skipped exactly
# (jnp.where), not added as zeros.
# =============================================================================


def k2v1_kernel_main(
    # Scalar prefetch
    c_flat_ref: jax.Array,  # int32[EP*EP*E_local] — expanded counts C
    d_flat_ref: jax.Array,  # int32[EP*EP] — dedup counts Ddup[src, dst]
    my_id_ref: jax.Array,   # int32[1]
    # In
    y_local_ref: jax.Array,  # [EP*cap_rows, K2] HBM — K1's expanded slots
    w_wide_ref: jax.Array,   # [EP*cap_rows, num_lanes] f32 HBM — per-expanded-
    # row weights, lane-replicated (epilogue scale sidecar; self slot patched
    # at the XLA level, so ALL slots are valid)
    rev_send_ref: jax.Array,  # int32[topk*EP*cap_dedup + PAD] HBM — HOME rev
    # buffer (self segment's reduce reads it; mirror of K1's self direct read)
    rev_recv_ref: jax.Array,  # int32[EP*topk*cap_dedup + PAD] HBM — K1-v1's
    # rev recv output, passed through
    rhs_ref: WeightsRef,      # weight [E_local, K2, N2] HBM
    # Out
    y_home_ref: jax.Array,    # [EP*cap_dedup, aligned_n] HBM — MY dedup'd
    # partial rows back, BLK-aligned segments per dst (aligned Ddup[my, d])
    y_stage_ref: jax.Array,   # [EP*cap_rows, aligned_n] HBM — per-src wo-gmm
    # staging (discarded output; zero_init'd per slot — the pad-row source)
    p_stage_ref: jax.Array,   # [EP*cap_dedup, aligned_n] HBM — per-src partial
    # staging for the return sends (discarded output)
    # Scratch
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    zero_ref: jax.Array,
    zero_sem_ref: jax.Array,
    send_sem: jax.Array,
    recv_sem: jax.Array,
    g_u32_ref: jax.Array,     # VMEM [tt, aligned_n] uint32 (gather staging)
    p_vmem_ref: jax.Array,    # VMEM [tt, aligned_n] bf16 (partial tile)
    rev_smem_ref: jax.Array,  # SMEM [topk*tt] int32
    gsem_ref: jax.Array,      # DMA sem [1]
    *,
    cfgs: GmmConfigs,  # per-segment EXPANDED configs, lhs_row_scale=True
    ep: int,
    cap_rows: int,
    cap_dedup: int,
    topk: int,
    blk: int,
    tt: int,           # dedup slots per reduce tile (static)
    device_id_fn=None,
    mode: str,
):
  """K2-v1 body: barrier -> EP x (scaled wo-gmm -> gather-reduce -> send) ->
  drains. Consume order / semaphore discipline identical to v0 k2_kernel_main."""
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

  width = y_home_ref.shape[-1]  # aligned_n
  out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
  ch3 = blk // out_sublane
  tt3 = tt // out_sublane
  y_home3 = y_home_ref.reshape(-1, out_sublane, width)
  y_stage3 = y_stage_ref.reshape(-1, out_sublane, width)
  p_stage3 = p_stage_ref.reshape(-1, out_sublane, width)
  ys_u32 = y_stage_ref.bitcast(jnp.uint32)  # [EP*cap_rows//2, width]
  lhs_in = y_local_ref.reshape(-1, sublane, size_k)
  scale_w = w_wide_ref.shape[-1]
  w_wide3 = w_wide_ref.reshape(-1, sublane, scale_w)
  cap3 = cap_rows // sublane

  # ---- 1. Barrier
  if mode != "nosend":
    bsem = pltpu.get_barrier_semaphore()
    for d in range(ep):
      pl.semaphore_signal(
          bsem, inc=1, device_id=device_id_fn(d), device_id_type=DEVICE_MESH
      )
    pl.semaphore_wait(bsem, ep)

  def _ret_off3(src):
    """MY aligned dedup segment start inside src's y_home (out tile-rows).

    = sum_{d' < my} align_up(Ddup[src, d'], BLK) — the v1 mirror of v0's
    _ret_off3, driven by the D table instead of C.
    """
    off = jnp.int32(0)
    for dp in range(ep):
      nblk = pl.cdiv(d_flat_ref[src * ep + jnp.int32(dp)], blk)
      off = off + jnp.where(jnp.int32(dp) < my, nblk * ch3, jnp.int32(0))
    return off

  def _send_segment(src):
    """Blocked return send of segment src's DEDUP'd partial rows."""
    nblk = pl.cdiv(d_flat_ref[src * ep + my], blk)
    dst_off3 = _ret_off3(src)
    src_slot3 = src * (cap_dedup // out_sublane)

    def _send_blocks(i, carry, src=src, dst_off3=dst_off3,
                     src_slot3=src_slot3):
      pltpu.make_async_remote_copy(
          p_stage3.at[pl.ds(src_slot3 + i * ch3, ch3)],
          y_home3.at[pl.ds(dst_off3 + i * ch3, ch3)],
          send_sem.at[src],
          recv_sem.at[my],
          device_id=device_id_fn(src),
          device_id_type=DEVICE_MESH,
      ).start()
      return carry

    lax.fori_loop(0, nblk, _send_blocks, jnp.int32(0))

  def _drain_dma_sem(sem_view, nblocks):
    rows3 = nblocks * ch3
    pltpu.make_async_copy(
        p_stage3.at[pl.ds(0, rows3)],
        p_stage3.at[pl.ds(0, rows3)],
        sem_view,
    ).wait()

  # My own aligned dedup start (rows) in MY home buffer / MY rev_send buffer.
  my_dd_start = jnp.int32(0)
  for dp in range(ep):
    nblk = pl.cdiv(d_flat_ref[my * ep + jnp.int32(dp)], blk)
    my_dd_start = my_dd_start + jnp.where(
        jnp.int32(dp) < my, nblk * blk, jnp.int32(0)
    )

  # ---- "nowait" probe: all return sends fire upfront (garbage staging).
  if mode == "nowait":
    for src_pos in range(1, ep):
      src = (my - src_pos + ep) % ep
      _send_segment(src)

  # ---- 2. Per-src: scaled gmm -> gather-reduce -> send (v0 consume order).
  gsem = gsem_ref.at[0]
  scratches = [partial_out_ref, acc_ref, metadata_ref]

  for src_pos in range(ep):
    src = (my - src_pos + ep) % ep
    base_c = (src * ep + my) * e_local

    if mode != "sendonly":
      num_gm = fill_metadata(
          c_flat_ref, None, metadata_ref, cfgs=cfgs, base=base_c
      )
      # gmm: lhs = y_local slot src, out = y_stage slot src, per-row weight
      # scale via the lhs_row_scale sidecar (w_wide slot src).
      if cfgs.zero_init:
        zero_size = zero_out_start(
            y_stage_ref, zero_ref, zero_sem_ref, metadata_ref, num_gm,
            cfgs=cfgs, row_offset=src * cap_rows, num_rows=cap_rows,
        )
      (lhs_spec, rhs_spec), out_spec = generate_block_specs(
          metadata_ref, cfgs, row_base=src * cap3
      )
      scale_spec = make_lhs_scale_spec(metadata_ref, cfgs, src * cap3)
      pipeline_fn = pltpu.emit_pipeline(
          functools.partial(inner_kernel_rs, cfgs=cfgs),
          grid=(num_n, num_gm, num_k),
          in_specs=(lhs_spec, rhs_spec, scale_spec),
          out_specs=out_spec,
      )
      pipeline_fn(lhs_in, rhs_ref, w_wide3, y_stage3, scratches=scratches)
      if cfgs.zero_init:
        zero_out_end(y_stage_ref, zero_sem_ref, zero_size, dims=cfgs.dims)

      # ---- gather-reduce: partial[s] = bf16 fold of the slot's <=topk
      # weighted rows (ascending j == ascending local expert id).
      n_slots = d_flat_ref[src * ep + my]
      y_row_base = src * cap_rows  # bf16 rows in y_stage
      if src_pos == 0:  # STATICALLY the self segment: partials go straight
        # into MY y_home at my aligned start; rev from MY send-side buffer.
        out3, out_base3 = y_home3, my_dd_start // out_sublane
        rev_ref, rev_base = rev_send_ref, my_dd_start * topk
      else:
        out3, out_base3 = p_stage3, src * (cap_dedup // out_sublane)
        rev_ref, rev_base = rev_recv_ref, src * (cap_dedup * topk)

      def _reduce_tile(i, carry, out3=out3, out_base3=out_base3,
                       rev_ref=rev_ref, rev_base=rev_base,
                       y_row_base=y_row_base):
        cp = pltpu.make_async_copy(
            rev_ref.at[pl.ds(rev_base + i * (tt * topk), tt * topk)],
            rev_smem_ref, gsem,
        )
        cp.start()
        cp.wait()

        def _row_abs(rr, jj):
          q = rev_smem_ref[rr * topk + jj]
          q = jnp.minimum(jnp.maximum(q, 0), cap_rows - 1)  # ALWAYS clamp
          return y_row_base + q

        acc_val = None
        for jj in range(topk):
          def _gcp(rr, jj=jj):
            a = _row_abs(rr, jj)
            return pltpu.make_async_copy(
                ys_u32.at[pl.ds(a // 2, 1)],
                g_u32_ref.at[pl.ds(rr, 1)],
                gsem,
            )

          for rr in range(tt):
            _gcp(rr).start()
          for rr in range(tt):
            _gcp(rr).wait()
          for rr in range(tt):
            shift = (_row_abs(rr, jj) & 1) * 16
            g_u32_ref[rr, :] = jnp.bitwise_and(
                g_u32_ref[rr, :] >> shift, 0xFFFF
            )
          term = jax.lax.bitcast_convert_type(
              g_u32_ref[...].astype(jnp.uint16), jnp.bfloat16
          )
          acc_val = term if acc_val is None else acc_val + term  # bf16, asc j

        p_vmem_ref[...] = acc_val
        ocp = pltpu.make_async_copy(
            p_vmem_ref.reshape(-1, out_sublane, width),
            out3.at[pl.ds(out_base3 + i * tt3, tt3)],
            gsem,
        )
        ocp.start()
        ocp.wait()
        return carry

      lax.fori_loop(0, pl.cdiv(n_slots, tt), _reduce_tile, jnp.int32(0))

    # Return send for THIS segment (dedup'd partial rows).
    if mode in ("full", "sendonly") and src_pos != 0:
      _send_segment(src)

  # ---- 3. Drains (v0 discipline, D-table byte counts).
  if mode != "nosend":
    for d in range(ep):
      nblk_send = jnp.where(
          my == jnp.int32(d),
          jnp.int32(0),
          pl.cdiv(d_flat_ref[jnp.int32(d) * ep + my], blk),
      )
      _drain_dma_sem(send_sem.at[d], nblk_send)
    for p in range(ep):
      nblk_recv = jnp.where(
          my == jnp.int32(p),
          jnp.int32(0),
          pl.cdiv(d_flat_ref[my * ep + jnp.int32(p)], blk),
      )
      _drain_dma_sem(recv_sem.at[p], nblk_recv)


def gmm_return_v1(
    y_local: jax.Array,   # [EP, cap_rows, K2] bf16 — K1-v1's expanded output
    counts_c: jax.Array,  # int32[EP, EP, E_local]
    dedup_d: jax.Array,   # int32[EP, EP]
    w_wide: jax.Array,    # [EP*cap_rows, num_lanes] f32 — per-expanded-row
    # weights, lane-replicated, ALL slots valid (prep.v1_w_wide of the PATCHED
    # side_recv — see prep_a2a.v1_patch_side_recv)
    rev_send: jax.Array,  # int32[topk*EP*cap_dedup + PAD] — home rev buffer
    rev_recv: jax.Array,  # int32[EP*topk*cap_dedup + PAD] — from gmm_a2a_v1
    rhs: jax.Array,       # [E_local, K2, N2] bf16 wo weights
    *,
    my_id: jax.Array,
    ep: int,
    cap_rows: int,
    cap_dedup: int,
    topk: int,
    collective_id: int = 47,
    blk: int = 512,
    tt: int = 128,        # dedup slots per reduce tile
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    mode: str = "full",
    ep_axis_name: str | None = None,
    mesh_axis_names: tuple | None = None,
    mesh_shape=None,
    interpret: bool = False,
) -> jax.Array:  # [EP*cap_dedup, N2] bf16 — MY dedup'd partial rows, BLK-
  # aligned segments per dst d (aligned Ddup[my, d] spans); slack undefined.
  """Fused v1 return: weight-scaled wo-gmm + per-slot gather-reduce +
  in-kernel a2a-back of DEDUP'd bf16 partials (WIRE_FORMAT_V1_SPEC §B)."""
  assert mode in ("full", "nosend", "sendonly", "nowait"), mode
  assert y_local.ndim == 3 and y_local.shape[0] == ep, y_local.shape
  assert y_local.shape[1] == cap_rows, (y_local.shape, cap_rows)
  size_k = y_local.shape[2]
  e_local, size_k2, size_n = rhs.shape
  assert size_k2 == size_k, (rhs.shape, size_k)
  assert counts_c.shape == (ep, ep, e_local), counts_c.shape
  assert dedup_d.shape == (ep, ep) and dedup_d.dtype == jnp.int32
  assert rev_send.shape == (topk * ep * cap_dedup + V1_SIDE_PAD,), rev_send.shape
  assert rev_recv.shape == (ep * topk * cap_dedup + V1_SIDE_PAD,), rev_recv.shape
  num_lanes = pltpu.get_tpu_info().num_lanes
  assert w_wide.shape == (ep * cap_rows, num_lanes) and w_wide.dtype == jnp.float32

  if tile_info is calculate_tiling and size_k == 2048:
    tile_info = TileSizes(tile_m=256, tile_k=2048, tile_n=3584)

  assert cap_rows % blk == 0 and cap_dedup % blk == 0
  assert cap_dedup % tt == 0, (cap_dedup, tt)
  assert (topk * tt) % 128 == 0 and (topk * blk) % 128 == 0

  cfgs, vmem_limit_bytes = k1_make_cfgs(
      y_local.dtype, rhs.dtype, size_k, size_n, e_local, cap_rows,
      blk=blk, tile_info=tile_info, vmem_limit_bytes=vmem_limit_bytes,
      lhs_row_scale=True,  # the per-expanded-row weight epilogue (w_wide)
  )
  dims = cfgs.dims
  tiles = cfgs.tiles
  assert cap_rows % dims.size_lhs_sublane == 0
  assert blk % dims.size_lhs_sublane == 0
  out_sublane = pltpu.get_tpu_info().get_sublane_tiling(cfgs.out_dtype)
  assert blk % out_sublane == 0 and tt % out_sublane == 0
  assert cap_rows % 2 == 0  # u32 pair-packing of y_stage

  aligned_n = align_to(cfgs.out_size_n, num_lanes)
  out_itemsize = jnp.dtype(cfgs.out_dtype).itemsize
  assert (blk * aligned_n * out_itemsize) % 128 == 0

  c_flat = counts_c.reshape(-1)
  d_flat = dedup_d.reshape(-1)
  my_id_arr = jnp.asarray(my_id, jnp.int32).reshape(1)
  y_local_flat = y_local.reshape(ep * cap_rows, size_k)

  target_zero_ref_bytes = 32 * 1024
  tile_zero_m = min(
      target_zero_ref_bytes // num_lanes // out_itemsize, dims.size_m
  )

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
      # V1 reduce scratch
      pltpu.VMEM((tt, aligned_n), jnp.uint32),
      pltpu.VMEM((tt, aligned_n), cfgs.out_dtype),
      pltpu.SMEM((topk * tt,), jnp.int32),
      pltpu.SemaphoreType.DMA((1,)),
  ]

  out_init = (
      jax.ShapeDtypeStruct((ep * cap_dedup, aligned_n), cfgs.out_dtype),
      jax.ShapeDtypeStruct((ep * cap_rows, aligned_n), cfgs.out_dtype),
      jax.ShapeDtypeStruct((ep * cap_dedup, aligned_n), cfgs.out_dtype),
  )
  rhs_weights = WeightsRef(weight=rhs, scale=None, bias=None)

  y_home, _y_stage, _p_stage = pl.pallas_call(
      functools.partial(
          k2v1_kernel_main, cfgs=cfgs, ep=ep, cap_rows=cap_rows,
          cap_dedup=cap_dedup, topk=topk, blk=blk, tt=tt,
          device_id_fn=make_device_id_fn(
              ep_axis_name, mesh_axis_names, mesh_shape
          ),
          mode=mode,
      ),
      out_shape=out_init,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=3,
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.HBM),  # y_local
              pl.BlockSpec(memory_space=pltpu.HBM),  # w_wide
              pl.BlockSpec(memory_space=pltpu.HBM),  # rev_send
              pl.BlockSpec(memory_space=pltpu.HBM),  # rev_recv
              WeightsRef(
                  weight=pl.BlockSpec(memory_space=pltpu.HBM),
                  scale=None, bias=None,
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
      name=f"k2v1_ret_ep{ep}_{mode}_blk{blk}-{get_scope_name(cfgs)}",
      cost_estimate=_k2_cost_estimate(cfgs, ep),
      metadata=get_metadata(cfgs),
      interpret=interpret,
  )(c_flat, d_flat, my_id_arr, y_local_flat, w_wide, rev_send, rev_recv,
    rhs_weights)

  return y_home[:, : cfgs.out_size_n]


def combine_home_v1(
    y_partial: jax.Array,  # [EP*cap_dedup, N2] bf16 — gmm_return_v1 output
    hit: jax.Array,        # int32[T, EP] — prep.build_v1_wire tables
    pos: jax.Array,        # int32[T, EP]
    d_counts: jax.Array,   # int32[EP] — Ddup[me, d]
    *,
    blk: int = 512,
) -> jax.Array:  # [T, N2] bf16
  """v1 home combine: out[t] = bf16 fold of its PRESENT partials, ascending
  src device d; absent dsts skipped exactly (jnp.where)."""
  t_tokens, ep = hit.shape
  dd_spans = (d_counts + blk - 1) // blk * blk
  dd_starts = jnp.cumsum(dd_spans) - dd_spans
  n_rows = y_partial.shape[0]
  acc = jnp.zeros((t_tokens, y_partial.shape[1]), y_partial.dtype)
  for d in range(ep):  # ascending src device — THE combine order contract
    rows = jnp.take(
        y_partial,
        jnp.clip(dd_starts[d] + pos[:, d], 0, n_rows - 1),
        axis=0,
    )
    acc = jnp.where((hit[:, d] > 0)[:, None], acc + rows, acc)
  return acc


# ----------------------------------------------------------------------------
# Home-side combine — NOT in the kernel, plain XLA in v0 (K2_KERNEL_SPEC.md).
# NOTE for later integration: this is where fp8-return dequant folds into the
# weights, and where the SC combine machinery can replace the XLA gather.
# ----------------------------------------------------------------------------


def combine_home(
    y_home: jax.Array,  # [M + EP*BLK, N2] — gmm_return output (BLK-aligned)
    send_perm: jax.Array,  # int32[M] — expanded (t*k+slot) -> send position
    weights: jax.Array,  # [T, k] router weights (any float dtype)
    send_sizes: jax.Array,  # int32[EP] — rows I sent each dst
    # (= counts.sum(axis=1), MY row of the C table)
    *,
    blk: int = 512,
) -> jax.Array:  # [T, N2] float32
  """Inverse-permutation gather from the aligned y_home + (T,k) f32 sum (§6).

  Maps each ragged send position p onto the BLK-aligned layout with the SAME
  align_up convention as gmm_return / K1's repack: segment d spans
  [al_starts[d], al_starts[d] + send_sizes[d]) where
  al_starts = cumsum(align_up(send_sizes, BLK)) (exclusive). Token-stationary;
  weights in f32, sum in f32 (cast to bf16 is the caller's epilogue).
  """
  t_tokens, k = weights.shape
  ep = send_sizes.shape[0]
  seg_starts = jnp.cumsum(send_sizes) - send_sizes  # ragged (prep_a2a math)
  al_spans = ((send_sizes + blk - 1) // blk) * blk
  al_starts = jnp.cumsum(al_spans) - al_spans  # aligned (K1 repack math)

  inv_perm = jnp.argsort(send_perm, stable=True)  # (t*k+slot) -> send position
  seg_of = jnp.zeros(inv_perm.shape, jnp.int32)
  for d in range(1, ep):  # static; empty segments resolve to the later one
    seg_of = seg_of + (inv_perm >= seg_starts[d]).astype(jnp.int32)
  al_pos = inv_perm - jnp.take(seg_starts, seg_of) + jnp.take(al_starts, seg_of)

  y_exp = jnp.take(y_home, al_pos, axis=0).reshape(t_tokens, k, -1)
  return jnp.sum(
      y_exp.astype(jnp.float32) * weights.astype(jnp.float32)[..., None],
      axis=1,
  )
