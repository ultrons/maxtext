# fused_layer_vjp.py — Phase D v0: custom_vjp for the fused a2a MoE layer
# (PHASE_D_SPEC.md). NO NEW PALLAS CODE — the backward is the forward kernels
# re-invoked with transposed weights (the kernel-reuse symmetry):
#
#   fwd:  x --prep--> x_send --K1(W1)--> y_local --K2(W2)--> y_home --comb(w)--> out
#   bwd:  dOut --comb-bwd--> dY_home --K1-KERNEL(W2^T, prealigned)--> dy_local
#              --K2-KERNEL(W1^T)--> dx_send_home --sort-bwd--> dx
#         dW1[e] += x_recv^T @ dy_local ; dW2[e] += y_local^T @ dz   (per src
#         segment, f32 accumulate; tgmm_v2 fork, imported not vendored)
#
# POSITION-MATH INVARIANT (the silent-corruption trap): every home-side
# expand/gather uses the ALIGNED segment layout — segment d starts at
# sum_{d'<d} align_up(C[my,d'].sum(), BLK) — via ONE helper
# (_aligned_expand_pos), the exact math of k2_fused.combine_home. Mixing
# ragged (cumsum(send_sizes)) and aligned positions corrupts the tokens
# nearest segment boundaries; check_bwd.py's finite-diff gate perturbs those
# tokens deliberately.
#
# v0 residuals (memory be damned — spec §6): x_recv, y_local, y_home, prep
# tables, C. x_recv comes back from gmm_a2a(return_recv=True) with slot [my]
# patched at the XLA level (the self segment never travels — K1's self-segment
# direct read leaves that slot unwritten). Production recompute-via-redispatch
# is out of v0 scope.
#
# Deviation from the spec text (documented, not a redesign): deliverable 1
# says "add prealigned, no other kernel change". The residual list (x_recv)
# and the dW2 formula (needs dz = the RECEIVED bwd segments) both require the
# x_recv buffer that gmm_a2a computes but discards — so gmm_a2a also grew a
# `return_recv` WRAPPER flag (the pallas body is untouched; the flag only
# stops the wrapper from discarding an existing pallas output).
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "tokamax_fork")))

from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from .k1_fused import calculate_tiling, gmm_a2a, gmm_a2a_v1  # noqa: E402
from .k2_fused import combine_home, combine_home_v1, gmm_return, gmm_return_v1  # noqa: E402
from . import prep_a2a as prep  # noqa: E402
from . import quant_utils as qu  # noqa: E402  (fp8 forward: per-row lhs + per-expert rhs quant)

# tgmm fork: IMPORTED (not vendored), UNJITTED (a @jax.jit inside shard_map is
# the known 10x+ Pallas compile trap; jax.jit exposes the raw fn as
# __wrapped__ — verified present on jax 0.10.1).
from .tgmm_v2_smemfix import tgmm_v2 as _tgmm_v2_jitted  # noqa: E402

tgmm_v2_unjitted = _tgmm_v2_jitted.__wrapped__


def _aligned_layout(send_sizes, blk):
  """(ragged seg_starts, aligned al_starts) from MY per-dst send sizes.

  al_starts is K1's x_send repack / gmm_return's y_home convention:
  cumsum(align_up(send_sizes, BLK)) exclusive. SINGLE SOURCE for all backward
  position math (see header trap note).
  """
  seg_starts = jnp.cumsum(send_sizes) - send_sizes
  al_spans = ((send_sizes + blk - 1) // blk) * blk
  al_starts = jnp.cumsum(al_spans) - al_spans
  return seg_starts, al_starts


def _aligned_expand_pos(send_perm, seg_starts, al_starts, ep):
  """al_pos int32[M]: aligned y_home/dx_send_home row of expanded row m.

  Line-for-line the position math of k2_fused.combine_home (empty segments
  resolve to the later one — zero-width in both cumsums, so equivalent).
  """
  inv_perm = jnp.argsort(send_perm, stable=True)  # m -> ragged send position
  seg_of = jnp.zeros(inv_perm.shape, jnp.int32)
  for d in range(1, ep):  # static loop, int32 arithmetic
    seg_of = seg_of + (inv_perm >= seg_starts[d]).astype(jnp.int32)
  return inv_perm - jnp.take(seg_starts, seg_of) + jnp.take(al_starts, seg_of)


def _dw_expert_major(lhs3, rhs3, counts, num_groups, cap):
  """Fused dW over ALL EP source segments in ONE tgmm (vs the per-src loop).

  The per-src accumulate (`for src: dw += tgmm(lhs3[src], ...)`) materialises the
  full [G,K,N] f32 dW EP times + EP XLA adds -> measured ~5.5x the HBM traffic of a
  single tgmm (dW is HBM-bound on the output write). Here we gather the src-major
  staging into EXPERT-MAJOR order once and run a single tgmm.

  lhs3 [EP,cap,K] rhs3 [EP,cap,N], counts int32[EP,num_groups] (per-src per-expert
  row counts; rows within a src slot are expert-sorted, then padding to cap).
  Returns dW [num_groups, K, N] f32.

  NOTE: reorders the f32 reduction across sources -> matches the per-src path to f32
  tolerance (~1e-7 measured), NOT bit-exact. Legitimate associativity change; opt-in
  via dw_impl="tgmm_em" so the bit-exact per-src path stays the default.
  """
  ep = lhs3.shape[0]
  M = ep * cap
  K = lhs3.shape[-1]
  N = rhs3.shape[-1]
  lf = lhs3.reshape(M, K)
  rf = rhs3.reshape(M, N)
  real_count = counts.sum(1)                                   # [EP]
  src_of = jnp.arange(M, dtype=jnp.int32) // cap
  pos_of = jnp.arange(M, dtype=jnp.int32) % cap
  real = pos_of < real_count[src_of]
  csum = jnp.cumsum(counts, axis=1)                            # [EP,G] inclusive
  expert_of = jax.vmap(
      lambda s, p: jnp.searchsorted(csum[s], p, side="right")
  )(src_of, pos_of).astype(jnp.int32)
  # sort key = expert-major, src-minor, original-pos; padding sentinel > every real
  # key (max ~ (G-1)*M + (M-1)) so padding lands past gs_all.sum() and is ignored.
  big = jnp.int32(num_groups * M + M)
  key = jnp.where(real, expert_of * M + src_of * cap + pos_of, big)
  idx = jnp.argsort(key)                                       # [M] gather perm
  lem = lf[idx]
  rem = rf[idx]
  gs_all = counts.sum(0)                                       # [G] per-expert totals
  return tgmm_v2_unjitted(lem, rem, gs_all, num_groups,
                          preferred_element_type=jnp.float32)


def _dw_segment_dense(lhs, rhs, gs, num_groups):
  """v0 FALLBACK dW for one segment: masked dense einsum (XLA level).

  lhs [CAP, K], rhs [CAP, N], gs int32[G] -> [G, K, N] f32. Row r belongs to
  group g iff cum[g-1] <= r < cum[g]; rows beyond cum[-1] are excluded.
  Correctness-first; O(G*CAP*K*N) — test shapes only.
  """
  cum = jnp.cumsum(gs)
  row = jnp.arange(lhs.shape[0], dtype=jnp.int32)
  gid = jnp.zeros_like(row)
  for g in range(num_groups - 1):  # static
    gid = gid + (row >= cum[g]).astype(jnp.int32)
  valid = (row < cum[num_groups - 1]).astype(jnp.float32)
  onehot = (
      (gid[:, None] == jnp.arange(num_groups, dtype=jnp.int32)[None, :])
  ).astype(jnp.float32) * valid[:, None]
  return jnp.einsum(
      "mg,mk,mn->gkn",
      onehot,
      lhs.astype(jnp.float32),
      rhs.astype(jnp.float32),
  )


def make_fused_moe_layer(
    *,
    ep: int,
    num_experts: int,
    cap_rows: int,
    blk: int = 512,
    tiles1=calculate_tiling,  # K1 profile (D->F); reused by bwd's prealigned
    # gmm_a2a (dY_home [., D] @ W2^T [., D, F] — same K/N profile)
    tiles2=calculate_tiling,  # K2 profile (F->D); reused by bwd's gmm_return
    ep_axis: str = "ep",
    mesh_axis_names: tuple | None = None,  # MULTI-AXIS mesh device addressing:
    mesh_shape=None,  # pass mesh.axis_names + dict(zip(names, mesh.devices.shape))
    # for MaxText's real mesh; None => single-axis "ep" (rank,) (our gates).
    recompute_residuals: bool = True,  # True: save inputs, RECOMPUTE the big
    # activations in _bwd (61L-memory-safe, but +~26% bwd compile — the recompute
    # re-emits 2 fwd kernels; see COMPILE_SCALING_NOTE.md). False: SAVE
    # x_recv/y_local/y_home (fast compile, but ~11GB/layer residuals). Pick by
    # scale: 25L fits memory -> False (fast compile); 61L -> True (memory).
    collective_ids=(42, 43, 44, 45),  # K1-fwd, K2-fwd, K1-kernel-bwd,
    # K2-kernel-bwd — distinct so all four barriers coexist in one program
    dw_impl: str = "tgmm",  # "tgmm" (imported fork) | "einsum" (v0 fallback)
    fp8_fwd: bool = False,  # CONVERGENCE-experiment path: fp8 FORWARD (e4m3 a2a wire
    # + fp8 gmms) with a straight-through BF16 backward (recompute + bwd unchanged).
    # The forward output reflects fp8 numerics; grads are the bf16 STE (bit-identical
    # to the bf16 layer). In-layer DYNAMIC quant (per-row lhs, per-expert rhs) — a
    # first-cut; production consumes external aqt/qwix pre-quantized weights. NOTE:
    # runs correctly but may be SLOWER (the fused K1 tile-guard/sidecar regression is
    # not fixed here — that's the separate PERFORMANCE track).
    self_last: bool = False,  # K2 OVERLAP FIX: compute the self (non-travelling) segment
    # LAST in gmm_return so the return a2a hides better. MEASURED bit-exact + combine
    # exposed wire 6.06 -> 2.38 ms/device (2.5x) at 4x8x8. Threaded to all gmm_return
    # calls (fwd/recompute/bwd). Default False = current behaviour; set True to enable.
):
  """Returns fused_moe_layer(x, indices, topk_weights, w1, w2) with custom_vjp.

  Runs INSIDE shard_map over `ep_axis` (check_vma=False). Shapes (per device):
  x [T, D] bf16; indices int32[T, k] GLOBAL expert ids; topk_weights [T, k]
  f32; w1 [E_local, D, F]; w2 [E_local, F, D]. Output [T, D] f32 (combine's
  f32 epilogue — exactly check_k2.fused_per_device's chain).
  """
  assert dw_impl in ("tgmm", "tgmm_em", "einsum"), dw_impl
  e_local = num_experts // ep
  cid1, cid2, cid3, cid4 = collective_ids
  _mkw = dict(ep_axis_name=ep_axis, mesh_axis_names=mesh_axis_names,
              mesh_shape=mesh_shape)  # threaded into every kernel call (fwd+bwd)

  def _forward(x, indices, weights, w1, w2, *, with_residuals: bool):
    k = indices.shape[1]
    my = lax.axis_index(ep_axis)
    send_perm, _ = prep.expand_and_sort(indices)
    x_send = prep.build_x_send(x, send_perm, k)
    counts = prep.build_counts(indices, num_experts, ep)
    c_table = lax.all_gather(counts, axis_name=ep_axis)  # [EPsrc, EPdst, El]

    # save_acts: SAVE x_recv/y_local/y_home (recompute_residuals=False path);
    # else the forward skips x_recv entirely (recomputed in _bwd).
    save_acts = with_residuals and not recompute_residuals
    if fp8_fwd:
      # fp8 dispatch: quantize activation per-row (e4m3 + scale sidecar on the wire)
      # + weights per-expert; the bwd stays bf16 (STE) so requires the recompute path.
      assert not save_acts, "fp8_fwd requires recompute_residuals=True (bf16 STE bwd)"
      xq, xs = qu.quantize_rows(x_send)                        # e4m3 [M,D], f32 [M]
      w1q, w1s = qu.quantize_weights(w1, block_k=w1.shape[1])  # per-expert (1 K-block)
      lhs1, w1_k, kw1 = xq, w1q, dict(lhs_row_scale=xs, rhs_scale=w1s)
    else:
      lhs1, w1_k, kw1 = x_send, w1, {}
    r1 = gmm_a2a(
        lhs1, c_table, w1_k, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows, blk=blk,
        tile_info=tiles1, collective_id=cid1, return_recv=save_acts, **kw1,
    )
    if save_acts:
      y_local, x_recv = r1
      # Patch x_recv slot [my] (self-segment direct read — never written).
      send_sizes = counts.sum(axis=1)
      seg_starts, _ = _aligned_layout(send_sizes, blk)
      x_send_pad = jnp.pad(x_send, ((0, cap_rows), (0, 0)))
      seg_my = lax.dynamic_slice_in_dim(
          x_send_pad, jnp.take(seg_starts, my), cap_rows
      )
      x_recv = lax.dynamic_update_slice(
          x_recv, seg_my[None], (my, jnp.int32(0), jnp.int32(0))
      )
    else:
      y_local, x_recv = r1, None

    if fp8_fwd:
      w2q, w2s = qu.quantize_weights(w2, block_k=w2.shape[1])  # per-expert
      w2_k, kw2 = w2q, dict(rhs_scale=w2s)  # y_local bf16 -> in-kernel lhs quant
    else:
      w2_k, kw2 = w2, {}
    y_home = gmm_return(
        y_local, c_table, w2_k, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows,
        m_rows=x_send.shape[0], blk=blk, tile_info=tiles2, collective_id=cid2,
        self_last=self_last, **kw2,
    )
    out = combine_home(
        y_home, send_perm, weights, counts.sum(axis=1), blk=blk
    )  # [T, D] f32
    # RESIDUAL REDUCTION (P2, 2026-07-13): save only the CHEAP inputs + int
    # routing; the ~11GB/layer activations (x_recv/y_local/y_home) are
    # RECOMPUTED in _bwd via _recompute_activations. nn.scan stacks residuals
    # across layers, so saving the big CAP=M buffers = the 61L OOM wall. Recompute
    # is bit-exact (same deterministic forward on the same inputs); costs ~1 extra
    # forward-dispatch in the bwd (cheap vs 60x memory). x is ~k*EP smaller than
    # x_recv (the dispatched/expanded rows). See P2_BUFFER_REUSE_SPEC.md.
    res = (
        (send_perm, c_table, x_recv, y_local, y_home, weights, w1, w2)
        if save_acts
        else (x, send_perm, c_table, weights, w1, w2)
    )
    return out, res

  def _recompute_activations(x, send_perm, c_table, w1, w2, k):
    """Bit-exact re-run of _forward's dispatch+project -> (x_recv, y_local,
    y_home). Mirrors _forward EXACTLY (same cids/tiles/patch) — the recompute
    half of the save-inputs residual policy."""
    my = lax.axis_index(ep_axis)
    x_send = prep.build_x_send(x, send_perm, k)
    y_local, x_recv = gmm_a2a(
        x_send, c_table, w1, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows, blk=blk,
        tile_info=tiles1, collective_id=cid1, return_recv=True,
    )
    counts_my = jnp.take(c_table, my, axis=0)
    send_sizes = counts_my.sum(axis=1)
    seg_starts, _ = _aligned_layout(send_sizes, blk)
    x_send_pad = jnp.pad(x_send, ((0, cap_rows), (0, 0)))
    seg_my = lax.dynamic_slice_in_dim(
        x_send_pad, jnp.take(seg_starts, my), cap_rows
    )
    x_recv = lax.dynamic_update_slice(
        x_recv, seg_my[None], (my, jnp.int32(0), jnp.int32(0))
    )
    y_home = gmm_return(
        y_local, c_table, w2, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows,
        m_rows=x_send.shape[0], blk=blk, tile_info=tiles2, collective_id=cid2,
        self_last=self_last,
    )
    return x_recv, y_local, y_home

  @jax.custom_vjp
  def fused_moe_layer(x, indices, weights, w1, w2):
    return _forward(x, indices, weights, w1, w2, with_residuals=False)[0]

  def _fwd(x, indices, weights, w1, w2):
    return _forward(x, indices, weights, w1, w2, with_residuals=True)

  def _bwd(res, dout):
    if recompute_residuals:
      x, send_perm, c_table, weights, w1, w2 = res
      t_tokens, k = weights.shape
      x_recv, y_local, y_home = _recompute_activations(
          x, send_perm, c_table, w1, w2, k
      )
    else:
      send_perm, c_table, x_recv, y_local, y_home, weights, w1, w2 = res
      t_tokens, k = weights.shape
    m_rows = t_tokens * k
    m_al = m_rows + ep * blk
    my = lax.axis_index(ep_axis)
    counts_my = jnp.take(c_table, my, axis=0)  # [EP, E_local] — MY C row
    send_sizes = counts_my.sum(axis=1)  # [EP]
    seg_starts, al_starts = _aligned_layout(send_sizes, blk)
    dout_f = dout.astype(jnp.float32)  # [T, D]

    # ---- 1. combine bwd (home, dense; ALIGNED positions throughout) ----
    al_pos = _aligned_expand_pos(send_perm, seg_starts, al_starts, ep)  # [M]
    y_exp = (
        jnp.take(y_home, al_pos, axis=0)
        .reshape(t_tokens, k, -1)
        .astype(jnp.float32)
    )
    d_weights = jnp.einsum("tkn,tn->tk", y_exp, dout_f).astype(weights.dtype)

    # dY_home in the ALIGNED send layout, GATHER form (row-stationary; no
    # scatter): aligned row p <- w'[m(p)] * dOut[token(m(p))], zero on slack.
    p = jnp.arange(m_al, dtype=jnp.int32)
    seg_of_p = jnp.zeros((m_al,), jnp.int32)
    for d in range(1, ep):  # static loop
      seg_of_p = seg_of_p + (p >= al_starts[d]).astype(jnp.int32)
    off = p - jnp.take(al_starts, seg_of_p)  # within aligned segment
    rag_q = off + jnp.take(seg_starts, seg_of_p)  # ragged send position
    in_fill = (off < jnp.take(send_sizes, seg_of_p)).astype(jnp.float32)
    mm = jnp.take(send_perm, jnp.clip(rag_q, 0, m_rows - 1))  # expanded row
    w_flat = weights.reshape(-1).astype(jnp.float32)
    scale = in_fill * jnp.take(w_flat, mm)  # [m_al]
    d_y_home = (scale[:, None] * jnp.take(dout_f, mm // k, axis=0)).astype(
        y_home.dtype
    )  # [m_al, D] bf16 — the bwd "wire" rows

    # ---- 2. K2 bwd (dx path) = the K1 KERNEL, W2 transposed, prealigned ----
    w2t = jnp.swapaxes(w2, 1, 2)  # [E_local, D, F]
    dy_local, dz = gmm_a2a(
        d_y_home, c_table, w2t, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows, blk=blk,
        tile_info=tiles1, collective_id=cid3, prealigned=True,
        return_recv=True,
    )  # dy_local [EP, CAP, F]; dz = received dY segments (slot my unwritten)
    # Patch dz slot [my]: my own segment of d_y_home at MY ALIGNED start.
    dY_pad = jnp.pad(d_y_home, ((0, cap_rows), (0, 0)))
    dz_my = lax.dynamic_slice_in_dim(
        dY_pad, jnp.take(al_starts, my), cap_rows
    )
    dz = lax.dynamic_update_slice(
        dz, dz_my[None], (my, jnp.int32(0), jnp.int32(0))
    )

    # ---- 3. K1 bwd (dx path) = the K2 KERNEL, W1 transposed ----
    w1t = jnp.swapaxes(w1, 1, 2)  # [E_local, F, D]
    dx_send_home = gmm_return(
        dy_local, c_table, w1t, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows,
        m_rows=m_rows, blk=blk, tile_info=tiles2, collective_id=cid4,
        self_last=self_last,
    )  # [m_al, D] — dx per expanded row, MY aligned send layout

    # ---- 4. sort bwd: inverse gather (SAME al_pos) + (T, k) segment sum ----
    dx = (
        jnp.take(dx_send_home, al_pos, axis=0)
        .reshape(t_tokens, k, -1)
        .astype(jnp.float32)
        .sum(axis=1)
        .astype(x_recv.dtype)  # == x.dtype (x_send is a row-gather of x)
    )

    # ---- 5. dW: per src segment, f32 accumulate across the EP segments ----
    if dw_impl == "tgmm_em":
      # ONE tgmm over expert-major-gathered rows (vs EP per-src tgmm + adds).
      # ~5.5x less HBM traffic on the dW; ~1e-7 vs the per-src path (reorder).
      C = jnp.stack([jnp.take(c_table[src], my, axis=0) for src in range(ep)])
      dw1 = _dw_expert_major(x_recv, dy_local, C, e_local, cap_rows)
      dw2 = _dw_expert_major(y_local, dz, C, e_local, cap_rows)
    else:
      dw1 = jnp.zeros(w1.shape, jnp.float32)  # [E_local, D, F]
      dw2 = jnp.zeros(w2.shape, jnp.float32)  # [E_local, F, D]
      for src in range(ep):  # static; accumulation ORDER = src 0..EP-1
        gs = jnp.take(c_table[src], my, axis=0)  # C[src, my, :] int32[E_local]
        if dw_impl == "tgmm":
          dw1 = dw1 + tgmm_v2_unjitted(
              x_recv[src], dy_local[src], gs, e_local,
              preferred_element_type=jnp.float32,
          )
          dw2 = dw2 + tgmm_v2_unjitted(
              y_local[src], dz[src], gs, e_local,
              preferred_element_type=jnp.float32,
          )
        else:
          dw1 = dw1 + _dw_segment_dense(x_recv[src], dy_local[src], gs, e_local)
          dw2 = dw2 + _dw_segment_dense(y_local[src], dz[src], gs, e_local)

    d_indices = np.zeros((t_tokens, k), jax.dtypes.float0)  # int arg
    return (
        dx,
        d_indices,
        d_weights,
        dw1.astype(w1.dtype),
        dw2.astype(w2.dtype),
    )

  fused_moe_layer.defvjp(_fwd, _bwd)
  return fused_moe_layer


# =============================================================================
# WIRE-FORMAT V1 layer (WIRE_FORMAT_V1_SPEC.md §C): dedup dispatch + weighted
# bf16 partial return, custom_vjp.
#
# BACKWARD STRUCTURE (spec C: "then the v0 machinery applies"): the composite
# transpose of {dedup-dispatch + in-gmm re-expansion} is IDENTICAL to v0's
# expanded transpose (expand-copy^T = sum over duplicates = the k-slot home
# sum v0 already does), so the backward WIRE stays v0-EXPANDED — both bwd
# a2a's reuse the UNMODIFIED v0 kernels. Deltas vs the v0 backward:
#   * The bwd dispatch ships UNWEIGHTED dOut rows (scale = in_fill only);
#     the expert multiplies by its per-expanded-row weight w_exp (the fwd
#     sidecar residual) AFTER the W2^T gmm. This yields, at the expert, BOTH
#     dy_local = w*u AND the unweighted u needed for d_topk_weights.
#   * d_topk_weights[t,e] = <y2[p], dOut[t]> = <y_local[p], u[p]> (y2 = wo-gmm
#     output never exists at home in v1). The per-expanded-row dots are
#     computed at the EXPERT and routed home via all_gather+slice — the
#     prep_a2a reference transport (bit-exact placement, no speed goal; an
#     in-kernel scalar return is future work, documented deviation).
#   * dW1's tgmm lhs is re-expanded at XLA from the dedup x_recv residual +
#     the index sidecar (x_exp = take(x_recv_dd, idx_exp)).
#   * blk_bwd (default = blk) sizes the v0-expanded bwd machinery separately:
#     the v0 kernels require m_rows % blk == 0, which the v1 wire (blk >= 64)
#     does not — tiny gate shapes pass blk_bwd < blk.
# =============================================================================


def make_fused_moe_layer_v1(
    *,
    ep: int,
    num_experts: int,
    cap_rows: int,        # EXPANDED slot rows: align_up(T*k, blk), %blk_bwd too
    blk: int = 512,       # v1 wire block (>= 64: 1-D int32 stream granularity)
    blk_bwd: int | None = None,  # v0-expanded bwd machinery block (m_rows %)
    tiles1=calculate_tiling,
    tiles2=calculate_tiling,
    ep_axis: str = "ep",
    mesh_axis_names: tuple | None = None,  # MULTI-AXIS mesh device addressing:
    mesh_shape=None,  # pass mesh.axis_names + dict(zip(names, mesh.devices.shape))
    # for MaxText's real mesh; None => single-axis "ep" (rank,) (our gates).
    collective_ids=(46, 47, 44, 45),  # K1-v1, K2-v1, bwd-K1-kernel, bwd-K2-kernel
    dw_impl: str = "tgmm",
):
  """Returns fused_moe_layer_v1(x, indices, topk_weights, w1, w2) w/ custom_vjp.

  Same call contract as make_fused_moe_layer; output is bf16 [T, D] (the v1
  combine is a bf16 fold — v0 returned the f32 combine epilogue)."""
  assert dw_impl in ("tgmm", "tgmm_em", "einsum"), dw_impl
  e_local = num_experts // ep
  cid1, cid2, cid3, cid4 = collective_ids
  _mkw = dict(ep_axis_name=ep_axis, mesh_axis_names=mesh_axis_names,
              mesh_shape=mesh_shape)  # threaded into every kernel call (fwd+bwd)
  if blk_bwd is None:
    blk_bwd = blk
  assert cap_rows % blk == 0 and cap_rows % blk_bwd == 0

  def _forward(x, indices, weights, w1, w2, *, with_residuals: bool):
    k = indices.shape[1]
    my = lax.axis_index(ep_axis)
    wire = prep.build_v1_wire(
        x, indices, weights, ep=ep, num_experts=num_experts, blk=blk
    )
    cap_dedup = wire["cap_dedup"]
    c_table = lax.all_gather(wire["counts"], axis_name=ep_axis)
    d_table = lax.all_gather(wire["d_counts"], axis_name=ep_axis)

    # x_recv_dd is NOT needed by the forward (out comes from y_ph) NOR saved
    # (residual reduction — recomputed in _bwd via _recompute_v1). side_recv/
    # rev_recv ARE returned regardless of return_recv and ARE needed (side_p,
    # gmm_return_v1), so keep them.
    del with_residuals
    y_local, side_recv, rev_recv = gmm_a2a_v1(
        wire["x_dedup"], c_table, d_table, wire["side_int"], wire["rev_int"],
        w1, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows, cap_dedup=cap_dedup,
        topk=k, blk=blk, tile_info=tiles1, collective_id=cid1,
        return_recv=False,
    )

    side_p = prep.v1_patch_side_recv(
        side_recv, wire["side_int"], wire["counts"], my,
        ep=ep, cap_rows=cap_rows, blk=blk,
    )
    num_lanes = pltpu.get_tpu_info().num_lanes
    w_wide = prep.v1_w_wide(
        side_p, ep=ep, cap_rows=cap_rows, num_lanes=num_lanes
    )
    y_ph = gmm_return_v1(
        y_local, c_table, d_table, w_wide, wire["rev_int"], rev_recv, w2,
        my_id=my, ep=ep, **_mkw, cap_rows=cap_rows, cap_dedup=cap_dedup, topk=k,
        blk=blk, tile_info=tiles2, collective_id=cid2,
    )
    out = combine_home_v1(
        y_ph, wire["hit"], wire["pos"], wire["d_counts"], blk=blk
    )  # [T, D] bf16
    # RESIDUAL REDUCTION (P2, 2026-07-13): save inputs + int routing only; the
    # big activations (x_recv_dd, y_local) + side_p are RECOMPUTED in _bwd via
    # _recompute_v1 (mirrors v0's policy; the 61L OOM fix — nn.scan stacks res).
    res = (x, indices, weights, w1, w2)
    return out, res

  def _recompute_v1(x, indices, weights, w1):
    """Bit-exact re-run of v1 _forward's wire-build + dispatch ->
    (send_perm, c_table, side_p, x_recv_dd, y_local). Mirrors _forward EXACTLY
    (same wire, cid1, tiles, x_recv_dd patch, side_p patch)."""
    my = lax.axis_index(ep_axis)
    k = indices.shape[1]
    wire = prep.build_v1_wire(
        x, indices, weights, ep=ep, num_experts=num_experts, blk=blk
    )
    cap_dedup = wire["cap_dedup"]
    c_table = lax.all_gather(wire["counts"], axis_name=ep_axis)
    d_table = lax.all_gather(wire["d_counts"], axis_name=ep_axis)
    y_local, side_recv, rev_recv, x_recv_dd = gmm_a2a_v1(
        wire["x_dedup"], c_table, d_table, wire["side_int"], wire["rev_int"],
        w1, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows, cap_dedup=cap_dedup,
        topk=k, blk=blk, tile_info=tiles1, collective_id=cid1, return_recv=True,
    )
    dd_spans = (wire["d_counts"] + blk - 1) // blk * blk
    dd_starts = jnp.cumsum(dd_spans) - dd_spans
    seg_my = lax.dynamic_slice_in_dim(
        wire["x_dedup"], jnp.take(dd_starts, my), cap_dedup
    )
    x_recv_dd = lax.dynamic_update_slice(
        x_recv_dd, seg_my[None], (my, jnp.int32(0), jnp.int32(0))
    )
    side_p = prep.v1_patch_side_recv(
        side_recv, wire["side_int"], wire["counts"], my,
        ep=ep, cap_rows=cap_rows, blk=blk,
    )
    return wire["send_perm"], c_table, side_p, x_recv_dd, y_local

  @jax.custom_vjp
  def fused_moe_layer_v1(x, indices, weights, w1, w2):
    return _forward(x, indices, weights, w1, w2, with_residuals=False)[0]

  def _fwd(x, indices, weights, w1, w2):
    return _forward(x, indices, weights, w1, w2, with_residuals=True)

  def _bwd(res, dout):
    x, indices, weights, w1, w2 = res
    send_perm, c_table, side_p, x_recv_dd, y_local = _recompute_v1(
        x, indices, weights, w1
    )
    t_tokens, k = weights.shape
    m_rows = t_tokens * k
    m_al = m_rows + ep * blk_bwd
    my = lax.axis_index(ep_axis)
    counts_my = jnp.take(c_table, my, axis=0)     # [EP, E_local]
    send_sizes = counts_my.sum(axis=1)
    seg_starts, al_starts = _aligned_layout(send_sizes, blk_bwd)
    dout_f = dout.astype(jnp.float32)

    # ---- 1. v1 combine+reduce bwd at home: UNWEIGHTED expanded dOut rows in
    # the v0 ALIGNED (blk_bwd) send layout. d(y2[p]) = w[p]*dOut[t(p)]; the w
    # factor is applied at the EXPERT (see header), so scale = in_fill only.
    p = jnp.arange(m_al, dtype=jnp.int32)
    seg_of_p = jnp.zeros((m_al,), jnp.int32)
    for d in range(1, ep):
      seg_of_p = seg_of_p + (p >= al_starts[d]).astype(jnp.int32)
    off = p - jnp.take(al_starts, seg_of_p)
    rag_q = off + jnp.take(seg_starts, seg_of_p)
    in_fill = (off < jnp.take(send_sizes, seg_of_p)).astype(jnp.float32)
    mm = jnp.take(send_perm, jnp.clip(rag_q, 0, m_rows - 1))
    d_y_unw = (in_fill[:, None] * jnp.take(dout_f, mm // k, axis=0)).astype(
        y_local.dtype
    )  # [m_al, D] bf16 — the (unweighted) bwd wire rows

    # ---- 2. bwd of the return path = the v0 K1 KERNEL, W2^T, prealigned ----
    w2t = jnp.swapaxes(w2, 1, 2)  # [E_local, D, F]
    u_local, dz_unw = gmm_a2a(
        d_y_unw, c_table, w2t, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows,
        blk=blk_bwd, tile_info=tiles1, collective_id=cid3, prealigned=True,
        return_recv=True,
    )  # u_local [EP, CAP, F] = UNWEIGHTED dOut @ W2^T per expanded row
    dY_pad = jnp.pad(d_y_unw, ((0, cap_rows), (0, 0)))
    dz_my = lax.dynamic_slice_in_dim(dY_pad, jnp.take(al_starts, my), cap_rows)
    dz_unw = lax.dynamic_update_slice(
        dz_unw, dz_my[None], (my, jnp.int32(0), jnp.int32(0))
    )

    # ---- 3. expert-side weight fold (fwd sidecar residual) ----
    idx_exp, w_exp = prep.v1_side_tables(side_p, ep=ep, cap_rows=cap_rows)
    dy_local = (u_local.astype(jnp.float32) * w_exp[:, :, None]).astype(
        y_local.dtype
    )
    dz_w = (dz_unw.astype(jnp.float32) * w_exp[:, :, None]).astype(
        y_local.dtype
    )

    # ---- 4. bwd of the dispatch = the v0 K2 KERNEL, W1^T; then sort-bwd ----
    w1t = jnp.swapaxes(w1, 1, 2)
    dx_send_home = gmm_return(
        dy_local, c_table, w1t, my_id=my, ep=ep, **_mkw, cap_rows=cap_rows,
        m_rows=m_rows, blk=blk_bwd, tile_info=tiles2, collective_id=cid4,
    )
    al_pos = _aligned_expand_pos(send_perm, seg_starts, al_starts, ep)
    dx = (
        jnp.take(dx_send_home, al_pos, axis=0)
        .reshape(t_tokens, k, -1)
        .astype(jnp.float32)
        .sum(axis=1)
        .astype(x_recv_dd.dtype)
    )

    # ---- 5. d_topk_weights: per-expanded-row <y_local, u> at the EXPERT,
    # routed home by all_gather+slice (reference transport — header note).
    dw_side = jnp.einsum(
        "scf,scf->sc", y_local.astype(jnp.float32),
        u_local.astype(jnp.float32),
    )  # [EP_src, CAP]
    dw_all = lax.all_gather(dw_side, axis_name=ep_axis)  # [EP_expert, EP_src, CAP]
    dw_rag = jnp.zeros((m_rows + cap_rows,), jnp.float32)
    for d in range(ep):  # my rows at expert d live in ITS src==my slot
      dw_rag = lax.dynamic_update_slice_in_dim(
          dw_rag, dw_all[d, my], jnp.take(seg_starts, d), axis=0
      )
    inv_perm = jnp.argsort(send_perm, stable=True)
    d_weights = (
        jnp.take(dw_rag, inv_perm).reshape(t_tokens, k).astype(weights.dtype)
    )

    # ---- 6. dW: per src segment, f32 accumulate (v0 order src 0..EP-1);
    # dW1's lhs re-expanded from the dedup residual via the index sidecar.
    cap_dedup = x_recv_dd.shape[1]
    x_exp_all = jnp.stack([  # re-expand the dedup residual per src -> [EP, cap, D]
        jnp.take(x_recv_dd[s], jnp.clip(idx_exp[s], 0, cap_dedup - 1), axis=0)
        for s in range(ep)
    ])
    if dw_impl == "tgmm_em":
      C = jnp.stack([jnp.take(c_table[src], my, axis=0) for src in range(ep)])
      dw1 = _dw_expert_major(x_exp_all, dy_local, C, e_local, cap_rows)
      dw2 = _dw_expert_major(y_local, dz_w, C, e_local, cap_rows)
    else:
      dw1 = jnp.zeros(w1.shape, jnp.float32)
      dw2 = jnp.zeros(w2.shape, jnp.float32)
      for src in range(ep):
        gs = jnp.take(c_table[src], my, axis=0)
        x_exp = x_exp_all[src]
        if dw_impl == "tgmm":
          dw1 = dw1 + tgmm_v2_unjitted(
              x_exp, dy_local[src], gs, e_local,
              preferred_element_type=jnp.float32,
          )
          dw2 = dw2 + tgmm_v2_unjitted(
              y_local[src], dz_w[src], gs, e_local,
              preferred_element_type=jnp.float32,
          )
        else:
          dw1 = dw1 + _dw_segment_dense(x_exp, dy_local[src], gs, e_local)
          dw2 = dw2 + _dw_segment_dense(y_local[src], dz_w[src], gs, e_local)

    d_indices = np.zeros((t_tokens, k), jax.dtypes.float0)
    return (
        dx,
        d_indices,
        d_weights,
        dw1.astype(w1.dtype),
        dw2.astype(w2.dtype),
    )

  fused_moe_layer_v1.defvjp(_fwd, _bwd)
  return fused_moe_layer_v1
