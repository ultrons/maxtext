# prep_a2a.py — WIRE_FORMAT.md v0: prep tables + unfused reference chain (L1-chain oracle).
#
# The reference implements the exact wire format (send order, counts table C, chunk-by-src
# gmm, slot-layout return, gather+dense combine) with plain XLA collectives (all_gather+slice
# stands in for the a2a — bit-exact row placement, no speed goal). The fused kernels must be
# BIT-IDENTICAL to this chain. Validated here against a dense per-token MoE oracle on the
# local 4x TPU v4 (EP=4), uniform + skewed routing.
#
# Run: ~/.venv-gmmfuse/bin/python prep_a2a.py
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P


# ----------------------------------------------------------------------------- prep (per device)
def expand_and_sort(indices):
  """Send permutation for WIRE_FORMAT §1.

  indices: int32[T, k] global expert ids.
  Returns (send_perm int32[M], e_sorted int32[M]) with M = T*k.
  Sorting by GLOBAL expert id IS (dst_device, local_expert) order, since
  dst = e // E_local and e % E_local orders within the dst segment.
  """
  e_flat = indices.reshape(-1)
  # stable argsort: jnp.argsort(stable=True)
  send_perm = jnp.argsort(e_flat, stable=True)
  return send_perm, e_flat[send_perm]


def build_counts(indices, num_experts, ep):
  """counts int32[EP, E_local]: rows I send to each dst, per dst-local expert (§2)."""
  e_local = num_experts // ep
  c = jnp.bincount(indices.reshape(-1), length=num_experts)
  return c.reshape(ep, e_local).astype(jnp.int32)


def build_x_send(x, send_perm, k):
  """x_send [M, D] — token rows expanded (row m carries x[send_perm[m] // k])."""
  return jnp.take(x, send_perm // k, axis=0)


# --------------------------------------------------------------- reference chain (inside shard_map)
def _segment_bounds(C, me):
  """Start of MY dst segment inside each src's send buffer + my recv sizes (prefix sums of §2)."""
  # C: [EP_src, EP_dst, E_local]
  per_dst = C.sum(axis=2)                      # [src, dst] rows src sends to dst
  start_in_src = jnp.cumsum(per_dst, axis=1) - per_dst   # exclusive prefix over dst
  return start_in_src[:, me], per_dst[:, me]   # [EP] starts, [EP] recv sizes


def reference_layer(x, indices, weights, w_local, *, ep, num_experts, ep_axis="ep"):
  """One MoE 'layer' (single gmm) in exact wire-format dataflow. Runs inside shard_map.

  x: [T, D] local tokens.  indices/weights: [T, k].  w_local: [E_local, D, F].
  Returns out [T, F] combined at home.
  """
  t_tokens, _ = x.shape
  k = indices.shape[1]
  m_rows = t_tokens * k
  me = jax.lax.axis_index(ep_axis)

  # prep + counts exchange (§2) — ONE table C feeds dispatch, return (and both bwds later)
  send_perm, _ = expand_and_sort(indices)
  x_send = build_x_send(x, send_perm, k)
  counts = build_counts(indices, num_experts, ep)
  C = jax.lax.all_gather(counts, axis_name=ep_axis)          # [EP_src, EP_dst, E_local]

  # ---- dispatch "a2a" (reference: all_gather + slice; fused kernel: remote DMAs) ----
  xs_all = jax.lax.all_gather(x_send, axis_name=ep_axis)     # [EP, M, D]
  xs_all = jnp.pad(xs_all, ((0, 0), (0, m_rows), (0, 0)))    # CAP-slice guard
  starts, recv_sizes = _segment_bounds(C, me)

  def recv_segment(src):
    return jax.lax.dynamic_slice_in_dim(xs_all[src], starts[src], m_rows, axis=0)

  x_recv = jnp.stack([recv_segment(s) for s in range(ep)])   # [EP, CAP=M, D] slot layout (§3)

  # ---- chunk-by-src gmm (§5): each segment already expert-sorted; group sizes = C[src, me] ----
  def seg_gmm(seg, gs):
    y = jax.lax.ragged_dot(seg, w_local, gs, preferred_element_type=jnp.float32)
    return y.astype(x.dtype)

  y_local = jnp.stack([seg_gmm(x_recv[s], C[s, me]) for s in range(ep)])  # [EP, CAP, F]

  # ---- return "a2a" (§6): exact reverse copy back into MY send order ----
  ys_all = jax.lax.all_gather(y_local, axis_name=ep_axis)    # [EP_dst, EP_src, CAP, F]
  send_sizes = C[me].sum(axis=1)                             # rows I sent to each dst
  my_starts = jnp.cumsum(send_sizes) - send_sizes            # my send-segment starts

  y_home = jnp.zeros((m_rows + m_rows, ys_all.shape[-1]), x.dtype)  # padded like xs_all
  for dst in range(ep):
    # dst computed my rows in ITS segment for src==me, rows [0 : send_sizes[dst])
    seg = ys_all[dst, me]                                    # [CAP, F]
    y_home = jax.lax.dynamic_update_slice_in_dim(y_home, seg, my_starts[dst], axis=0)
  y_home = y_home[:m_rows]

  # ---- combine (§6): local inverse gather + dense (T, k) weighted sum — no scatter ----
  inv_perm = jnp.argsort(send_perm, stable=True)             # m -> position in send order
  y_exp = jnp.take(y_home, inv_perm, axis=0).reshape(t_tokens, k, -1)
  return jnp.sum(y_exp.astype(jnp.float32) * weights.astype(jnp.float32)[..., None], axis=1)


# ============================================================================
# WIRE-FORMAT V1 (WIRE_FORMAT_V1_SPEC.md): home-side prep for the dedup
# dispatch + weighted bf16 partial return. All wire buffers are emitted
# ALIGNED directly (the Package-2 "prealigned-primary" direction — no
# wrapper-side repack in v1).
#
# Layout choices (documented per spec A.2/A.4 "implementer's choice"):
#   * The index sidecar (A.2) and weight sidecar (A.4) travel as ONE
#     INTERLEAVED int32 stream `side_int`: element 2p = dedup offset of
#     expanded row p (segment-relative), element 2p+1 = bitcast(f32 topk
#     weight of row p). Layout = v0's BLK-aligned EXPANDED segment layout
#     (segment d starts at 2*sum_{d'<d} align_up(C_rows_d', BLK)).
#   * A REVERSE sidecar `rev_int` (slot-major, k int32 per dedup slot, in the
#     dedup buffer's BLK-aligned segment layout) also ships with the segment:
#     rev[s*k + j] = segment-relative EXPANDED position of the j-th expanded
#     row of dedup slot s, ascending local expert id (ties by send position).
#     Pad entries (token has < k experts on the dst) point at expanded row
#     `fill` (the first zero_init'd slack row of the segment) so a padded
#     gather contributes an exact zero row to the bf16 partial sum. This is
#     what makes K2-v1's per-slot gather-reduce TC-legal with no scatter.
#   * Weights are NOT in rev: K2-v1 applies w per EXPANDED row inside the
#     wo-gmm epilogue (the lhs_row_scale machinery) using the w half of
#     side_int, so the reduce is a plain sum of already-weighted rows.
# ============================================================================

V1_SIDE_PAD = 128  # int32 slack after every sidecar buffer: the kernel's SMEM
# staging window is fetched 128-aligned and may over-read up to 128 elements.


def _align_up_i(n, a):
  return (n + a - 1) // a * a


def build_v1_wire(x, indices, weights, *, ep, num_experts, blk):
  """All home-side v1 wire buffers + dedup tables (per device, XLA level).

  x [T, D]; indices int32[T, k] global expert ids; weights [T, k].
  Returns a dict:
    send_perm  int32[M]       v0 send permutation (expanded)
    counts     int32[EP, E_local]  v0 counts (my row of C)
    hit        int32[T, EP]   1 iff token t has >=1 expert on dst d
    pos        int32[T, EP]   exclusive cumsum of hit (dedup slot of t in seg d)
    d_counts   int32[EP]      Ddup[me, d] = hit.sum(0)  (joins the C exchange)
    x_dedup    [EP*cap_dedup, D]  dedup rows, BLK-aligned segments (dst-major,
                              ascending token order within a segment)
    side_int   int32[2*(M+EP*BLK) + PAD]  interleaved (idx, w-bits) sidecar
    rev_int    int32[k*EP*cap_dedup + PAD] slot-major reverse sidecar
    cap_dedup  int (static)   align_up(T, BLK) — per-src dedup slot rows
  """
  t_tokens, k = indices.shape
  m_rows = t_tokens * k
  e_local = num_experts // ep
  e_flat = indices.reshape(-1)
  dst_flat = (e_flat // e_local).astype(jnp.int32)
  send_perm = jnp.argsort(e_flat, stable=True)
  counts = build_counts(indices, num_experts, ep)      # [EP, E_local]
  send_sizes = counts.sum(axis=1).astype(jnp.int32)    # [EP] expanded rows/dst
  seg_starts = jnp.cumsum(send_sizes) - send_sizes     # ragged expanded starts
  al_spans = (send_sizes + blk - 1) // blk * blk
  al_starts = jnp.cumsum(al_spans) - al_spans          # aligned expanded starts

  # ---- dedup tables (spec A.1)
  dst_tk = dst_flat.reshape(t_tokens, k)
  hit = jnp.stack(
      [(dst_tk == d).astype(jnp.int32).max(axis=1) for d in range(ep)], axis=1
  )                                                    # int32[T, EP]
  pos = jnp.cumsum(hit, axis=0) - hit                  # exclusive, per column
  d_counts = hit.sum(axis=0).astype(jnp.int32)         # [EP]
  cap_dedup = _align_up_i(t_tokens, blk)               # static
  m_dd_al = ep * cap_dedup                             # static buffer bound
  dd_spans = (d_counts + blk - 1) // blk * blk
  dd_starts = jnp.cumsum(dd_spans) - dd_spans          # aligned dedup starts

  # ---- dedup row buffer: token of each buffer row (hit tokens first, in
  # ascending token order — argsort(1-hit) stable); slack rows clamp to a
  # valid token (never read through a *valid* sidecar index).
  tok_order = jnp.stack(
      [jnp.argsort(1 - hit[:, d], stable=True) for d in range(ep)]
  )                                                    # [EP, T]
  r = jnp.arange(m_dd_al, dtype=jnp.int32)
  seg_of_r = jnp.zeros((m_dd_al,), jnp.int32)
  for d in range(1, ep):
    seg_of_r = seg_of_r + (r >= dd_starts[d]).astype(jnp.int32)
  q_r = jnp.clip(r - jnp.take(dd_starts, seg_of_r), 0, t_tokens - 1)
  tok_of_r = jnp.take(tok_order.reshape(-1), seg_of_r * t_tokens + q_r)
  x_dedup = jnp.take(x, tok_of_r, axis=0)              # [m_dd_al, D]

  # ---- expanded sidecar (A.2 + A.4, interleaved), v0-aligned layout
  tok_m = jnp.arange(m_rows, dtype=jnp.int32) // k
  idx_exp_m = jnp.take(pos.reshape(-1), tok_m * ep + dst_flat)   # [M]
  w_flat = weights.reshape(-1).astype(jnp.float32)
  side_idx_send = jnp.take(idx_exp_m, send_perm)
  side_w_send = jnp.take(w_flat, send_perm)
  m_al = m_rows + ep * blk
  ra = jnp.arange(m_al, dtype=jnp.int32)
  seg_of_a = jnp.zeros((m_al,), jnp.int32)
  for d in range(1, ep):
    seg_of_a = seg_of_a + (ra >= al_starts[d]).astype(jnp.int32)
  src_idx = ra - jnp.take(al_starts, seg_of_a) + jnp.take(seg_starts, seg_of_a)
  src_idx = jnp.clip(src_idx, 0, m_rows - 1)           # slack rows: valid copies
  side_idx_al = jnp.take(side_idx_send, src_idx)
  side_w_al = jnp.take(side_w_send, src_idx)
  side_int = jnp.stack(
      [side_idx_al, jax.lax.bitcast_convert_type(side_w_al, jnp.int32)], axis=1
  ).reshape(-1)                                        # [2*m_al] interleaved
  side_int = jnp.pad(side_int, (0, V1_SIDE_PAD))

  # ---- reverse sidecar (slot-major, k ints per dedup slot, dedup-aligned)
  g = tok_m * ep + dst_flat                            # (token, dst) group
  g_send = jnp.take(g, send_perm)                      # in send order p
  order2 = jnp.argsort(g_send, stable=True)            # groups; within: p asc
  g_sorted = jnp.take(g_send, order2)
  first = jnp.searchsorted(g_sorted, g_sorted, side="left").astype(jnp.int32)
  j_sorted = jnp.arange(m_rows, dtype=jnp.int32) - first
  j_send = jnp.zeros((m_rows,), jnp.int32).at[order2].set(j_sorted)
  p_send = jnp.arange(m_rows, dtype=jnp.int32)
  dst_send = jnp.take(dst_flat, send_perm)
  tok_send = jnp.take(tok_m, send_perm)
  p_rel = p_send - jnp.take(seg_starts, dst_send)      # ragged == y-slot row
  slot_row = jnp.take(dd_starts, dst_send) + jnp.take(
      pos.reshape(-1), tok_send * ep + dst_send
  )
  ent = slot_row * k + j_send
  # pads point at expanded row `fill` of the row's segment (zero_init'd slack;
  # pads exist only when fill < cap_rows — see the k2v1 invariant note).
  pad_idx = jnp.take(send_sizes, seg_of_r)             # [m_dd_al]
  rev = jnp.broadcast_to(pad_idx[:, None], (m_dd_al, k)).reshape(-1)
  rev_int = rev.at[ent].set(p_rel)
  rev_int = jnp.pad(rev_int, (0, V1_SIDE_PAD))

  return dict(
      send_perm=send_perm,
      counts=counts,
      hit=hit,
      pos=pos,
      d_counts=d_counts,
      x_dedup=x_dedup,
      side_int=side_int,
      rev_int=rev_int,
      cap_dedup=cap_dedup,
  )


# ---- v1 chain glue (XLA level, between K1-v1 and K2-v1 / for the backward)


def v1_exp_aligned_starts(counts, blk):
  """Aligned EXPANDED segment starts (v0 layout) from MY counts row."""
  send_sizes = counts.sum(axis=1).astype(jnp.int32)
  spans = (send_sizes + blk - 1) // blk * blk
  return jnp.cumsum(spans) - spans


def v1_patch_side_recv(side_recv, side_int, counts, my, *, ep, cap_rows, blk):
  """Fill side_recv slot [my] (never travels — K1's self-segment direct read)
  from the send-side sidecar at MY aligned expanded start. Needed before
  v1_w_wide (K2's epilogue scale) and as the backward's w_exp residual."""
  al = v1_exp_aligned_starts(counts, blk)
  src = jnp.pad(side_int, (0, 2 * cap_rows))
  seg = jax.lax.dynamic_slice_in_dim(src, 2 * jnp.take(al, my), 2 * cap_rows)
  return jax.lax.dynamic_update_slice_in_dim(
      side_recv, seg, my * (2 * cap_rows), axis=0
  )


def v1_side_tables(side_recv_p, *, ep, cap_rows):
  """(idx_exp int32[EP, cap_rows], w_exp f32[EP, cap_rows]) from the PATCHED
  side_recv. Rows beyond a segment's fill carry slack/garbage — every
  consumer bounds them by the C table."""
  ints = side_recv_p[: ep * 2 * cap_rows].reshape(ep, cap_rows, 2)
  return ints[..., 0], jax.lax.bitcast_convert_type(ints[..., 1], jnp.float32)


def v1_w_wide(side_recv_p, *, ep, cap_rows, num_lanes):
  """Lane-replicated per-expanded-row weight buffer for K2-v1's epilogue."""
  _, w_exp = v1_side_tables(side_recv_p, ep=ep, cap_rows=cap_rows)
  w = w_exp.reshape(ep * cap_rows)
  return jnp.broadcast_to(w[:, None], (ep * cap_rows, num_lanes))


# ----------------------------------------------------------------------------- dense oracle
def dense_oracle(x, indices, weights, w_global):
  """out[t] = sum_slot w * (x[t] @ W[e]) in f32 — trivially correct, O(T*k) weight gathers."""
  xf = x.astype(jnp.float32)
  wf = w_global.astype(jnp.float32)
  y_all = jnp.einsum("td,edf->tef", xf, wf)                     # [T, E, F] — small at test sizes
  y = jnp.take_along_axis(y_all, indices[..., None], axis=1)    # [T, k, F]
  return jnp.sum(y * weights.astype(jnp.float32)[..., None], axis=1)


# ----------------------------------------------------------------------------- test harness
def make_routing(rng, t_tokens, k, num_experts, skew_h=0.0):
  """Random top-k routing; skew_h redirects that fraction of slots to a hot-16 expert set."""
  idx = np.stack([rng.choice(num_experts, size=k, replace=False) for _ in range(t_tokens)])
  if skew_h > 0:
    hot = rng.integers(0, 16, size=idx.shape)
    idx = np.where(rng.random(idx.shape) < skew_h, hot, idx)
  w = rng.random((t_tokens, k)).astype(np.float32)
  w /= w.sum(axis=1, keepdims=True)
  return idx.astype(np.int32), w


def run_test(ep=4, t_tokens=512, d=256, f=512, num_experts=32, k=8, skew_h=0.0, seed=0):
  rng = np.random.default_rng(seed)
  e_local = num_experts // ep
  devs = jax.devices()[:ep]
  mesh = Mesh(np.array(devs), ("ep",))

  # per-device data (built globally, sharded over ep)
  x = (rng.standard_normal((ep * t_tokens, d)) * 0.1).astype(np.float32).astype(jnp.bfloat16)
  idx_list, w_list = zip(*[make_routing(rng, t_tokens, k, num_experts, skew_h) for _ in range(ep)])
  idx = np.concatenate(idx_list)
  wts = np.concatenate(w_list)
  w_global = (rng.standard_normal((num_experts, d, f)) * 0.05).astype(np.float32).astype(jnp.bfloat16)

  layer = functools.partial(reference_layer, ep=ep, num_experts=num_experts)
  fn = jax.jit(
      jax.shard_map(
          lambda xx, ii, ww, wl: layer(xx, ii, ww, wl[0]),  # wl arrives [1, E_local, D, F]
          mesh=mesh,
          in_specs=(P("ep"), P("ep"), P("ep"), P("ep")),
          out_specs=P("ep"),
          check_vma=False,
      )
  )
  w_sharded = w_global.reshape(ep, e_local, d, f)
  out = np.asarray(fn(x, idx, wts, w_sharded))
  ref = np.asarray(dense_oracle(jnp.asarray(x), jnp.asarray(idx), jnp.asarray(wts), jnp.asarray(w_global)))

  err = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9)
  ok = np.allclose(out, ref, rtol=3e-2, atol=3e-2)
  print(f"EP={ep} T={t_tokens} E={num_experts} k={k} skew={skew_h}: rel_max={err:.2e} allclose={ok}")
  return ok


if __name__ == "__main__":
  results = [
      run_test(skew_h=0.0),
      run_test(skew_h=0.65, seed=1),
      run_test(t_tokens=1024, num_experts=64, skew_h=0.3, seed=2),
  ]
  assert all(results), "wire-format reference chain FAILED vs dense oracle"
  print("ALL PASS — wire-format reference chain == dense oracle")
