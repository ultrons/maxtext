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

"""CPU equivalence test: chunked_ring_dispatch vs un-chunked all_gather + ring_ragged_sort.

Run:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
      python3 tests/chunked_dispatch_equiv_test.py

The INPUT-side dual of the combine test. The dispatch chunks the EP all-gather over the input
token axis. Route B (rung 9e, compaction-first): per chunk ONE COMPACTED gather into a
buffer_size/N piece (RANK-space bounds via searchsorted), then ONE full-buffer placement gather
(buffer-position bounds AS-IS). This gates:
  * fwd/bwd BIT-EXACT (diff == 0) vs the un-chunked all_gather + ring_ragged_sort reference;
  * numpy SC-addressing emulation of BOTH new calls: compaction block ranges in-bounds for the
    chunk-sized arrays with rank-space bounds (and OOB/wrong with raw buffer-position bounds --
    the contrast that proves the conversion is load-bearing), placement reads only WRITTEN piece
    rows for every shard row;
  * the chunk_local_row < xg_c.rows invariant (the one value the compaction random-accesses).
EP=8 + NON-UNIFORM group sizes is the catch case (uniform hides alignment bugs).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.ragged.ragged_sort import chunked_ring_dispatch, ring_ragged_sort

TOPK = 4
NUM_EXPERTS = 16
HIDDEN = 16
NUM_TOKENS_LOCAL = 32  # per-shard tokens; global = local * ep


def _make_topk(key, num_tokens_global, skew):
  logits = jax.random.normal(key, (num_tokens_global, NUM_EXPERTS))
  logits = logits + skew * jax.random.normal(jax.random.fold_in(key, 1), (NUM_EXPERTS,))
  return jax.lax.top_k(logits, TOPK)[1].astype(jnp.int32)  # [G, topk], distinct per token


# Forced pure-JAX fallback (no SparseCore on CPU; get_tpu_info raises on non-TPU).
FB = dict(enforce_gather_fallback=True, enforce_gather_reduce_fallback=True)


def _unchunked(mesh, ep_size, x_local, topk_global):
  """all_gather(x) then ring_ragged_sort -- the reference dispatch."""

  def body(xl, tg):
    xg = jax.lax.all_gather(xl, "ep", axis=0, tiled=True)  # [G, hidden]
    buf, gs, revert = ring_ragged_sort(
        xg, tg, NUM_EXPERTS, TOPK, "ep", ep_size, buffer_size=None, **FB
    )
    return buf

  fn = jax.shard_map(body, mesh=mesh, in_specs=(jax.P("ep"), jax.P()), out_specs=jax.P("ep"))
  return jax.jit(fn)(x_local, topk_global)


def _chunked(mesh, ep_size, x_local, topk_global, n):
  def body(xl, tg):
    buf, gs, revert = chunked_ring_dispatch(xl, tg, NUM_EXPERTS, TOPK, "ep", ep_size, n, **FB)
    return buf

  fn = jax.shard_map(body, mesh=mesh, in_specs=(jax.P("ep"), jax.P()), out_specs=jax.P("ep"))
  return jax.jit(fn)(x_local, topk_global)


def _addressing_invariant_check(failures):
  """SC-addressing invariant (bounds checks are OFF on SparseCore -> silent OOB).

  The dispatch feeds ``ragged_gather`` FULL-shape ``idx_c`` and FULL
  ``[shard_output_start, shard_output_end)`` bounds -- byte-identical to the un-chunked
  ``ring_ragged_sort`` call, so the kernel's row/partition COUNTS are unchanged (derived from the
  full index/output axis, not from the chunk-sized source ``xg_c``). The one chunk-dependent value
  the kernel random-accesses is ``chunk_local_row`` indexing ``xg_c`` (rows ``per*ep_size``): every
  such index MUST be < that, for the in-window rows of EVERY chunk, or the SC gather reads garbage.
  This numpy emulation of the index arithmetic asserts it across all shards / chunks.
  """
  topk = TOPK
  for ep_size in (8, 4):
    ntl = NUM_TOKENS_LOCAL
    ntg = ntl * ep_size
    topk_global = np.asarray(_make_topk(jax.random.PRNGKey(11 + ep_size), ntg, 3.0))
    flat = topk_global.reshape(-1)
    argsort = np.argsort(flat, kind="stable")
    token_indices = np.repeat(np.arange(ntg, dtype=np.int32), topk)
    token_indices_sorted = token_indices[argsort]
    gs = np.bincount(flat, minlength=NUM_EXPERTS)
    offs = np.concatenate([[0], np.cumsum(gs)])
    local_e = NUM_EXPERTS // ep_size
    for n_chunks in (1, 2, 4, 8):
      per = ntl // n_chunks
      origin_shard = token_indices_sorted // ntl
      local_idx = token_indices_sorted % ntl
      chunk_of_row = local_idx // per
      in_chunk_pos = local_idx - chunk_of_row * per
      chunk_local_row = origin_shard * per + in_chunk_pos
      xg_rows = per * ep_size
      bad = 0
      for s in range(ep_size):
        lo, hi = int(offs[s * local_e]), int(offs[(s + 1) * local_e])
        for c in range(n_chunks):
          rows = np.arange(lo, hi)
          sel = rows[chunk_of_row[lo:hi] == c]
          if sel.size and (chunk_local_row[sel].max() >= xg_rows or chunk_local_row[sel].min() < 0):
            bad += 1
      print(f"  addressing EP={ep_size} N={n_chunks}: out-of-range shard-chunks={bad}")
      if bad:
        failures.append(f"addressing EP={ep_size} N={n_chunks}: chunk_local_row OOB for xg_c ({bad})")


def _emulate_sc_gather(g, indices, start, end, lanes, num_cores):
  """Numpy emulation of the SC ragged_gather wrapper + main_kernel ADDRESSING (no weights).

  Mirrors: wrapper padding of indices to a block multiple, and main_kernel's block-range
  derivation ``block_start = start // block_size``, ``block_end = cdiv(end, block_size)`` which
  indexes the INDEX/OUTPUT row axis. Returns (out, oob): ``out`` has one row per index (NaN
  sentinel where the kernel never wrote -- uninitialized HBM on hardware), ``oob`` is True when
  the block range indexes the (possibly chunk-sized) index/output arrays out of bounds (silent
  on SC: bounds checks off).
  """
  n = indices.shape[0]
  bs = lanes * num_cores
  padded_n = -(-n // bs) * bs
  block_start = int(start) // bs
  block_end = -(-int(end) // bs)
  if int(end) == int(start):
    block_end = block_start
  lo_row, hi_row = block_start * bs, block_end * bs
  oob = hi_row > padded_n or lo_row > padded_n or lo_row < 0
  out = np.full((n, g.shape[1]), np.nan, np.float32)
  for r in range(max(lo_row, 0), min(hi_row, n)):
    out[r] = g[indices[r]]
  return out, oob


def _route_b_emulation_check(failures):
  """Emulated-SC addressing for BOTH Route-B gather calls (compaction + placement).

  Per shard s and chunk c:
    * COMPACTION: indices/output are chunk-sized (rows_per_chunk). With RANK-space bounds
      [searchsorted(rows_c, lo), searchsorted(rows_c, hi)) the block range must be in-bounds and
      the written rows exact; with raw BUFFER-POSITION bounds it must go OOB or write wrong rows
      for some shard/chunk at every N > 1 (proving the conversion is load-bearing).
    * PLACEMENT: full-length indices (n == buffer_size), buffer-position bounds AS-IS; every
      shard row j in [lo, hi) must read a piece position INSIDE its chunk's block-rounded
      WRITTEN window (never an unwritten/uninitialized piece row) and reproduce the reference
      value x_global[token_indices_sorted[j]].
  """
  lanes, num_cores = 8, 4  # block_size 32: small vs the chunk length so ranges have resolution
  topk = TOPK
  for ep_size in (8, 4):
    ntl = NUM_TOKENS_LOCAL
    ntg = ntl * ep_size
    key = jax.random.PRNGKey(21 + ep_size)
    topk_global = np.asarray(_make_topk(key, ntg, 3.0))
    x_global = np.asarray(jax.random.normal(jax.random.fold_in(key, 3), (ntg, HIDDEN), dtype=jnp.float32))
    flat = topk_global.reshape(-1)
    argsort = np.argsort(flat, kind="stable")
    token_indices = np.repeat(np.arange(ntg, dtype=np.int32), topk)
    tis = token_indices[argsort]  # buffer row -> global token
    gs = np.bincount(flat, minlength=NUM_EXPERTS)
    offs = np.concatenate([[0], np.cumsum(gs)])
    local_e = NUM_EXPERTS // ep_size
    buffer_size = ntg * topk
    for n_chunks in (1, 2, 4, 8):
      per = ntl // n_chunks
      origin_shard = tis // ntl
      local_idx = tis % ntl
      chunk_of_row = local_idx // per
      chunk_local_row = origin_shard * per + local_idx - chunk_of_row * per
      rpc = buffer_size // n_chunks
      order = np.argsort(chunk_of_row, kind="stable")
      piece_pos = np.argsort(order)
      src_all = chunk_local_row[order]
      ok_comp = True
      bad_buggy = 0
      total = 0
      ok_place = True
      for s in range(ep_size):
        lo, hi = int(offs[s * local_e]), int(offs[(s + 1) * local_e])
        if hi == lo:
          continue
        pieces = np.full((buffer_size, HIDDEN), np.nan, np.float32)
        bs = lanes * num_cores
        for c in range(n_chunks):
          total += 1
          rows_c = order[c * rpc : (c + 1) * rpc]
          src_c = src_all[c * rpc : (c + 1) * rpc]
          # chunk c's all-gathered x: shard-major rows within the chunk
          xg_c = np.concatenate([x_global[sh * ntl + c * per : sh * ntl + (c + 1) * per] for sh in range(ep_size)])
          cs = int(np.searchsorted(rows_c, lo))
          ce = int(np.searchsorted(rows_c, hi))
          piece, oob = _emulate_sc_gather(xg_c, src_c, cs, ce, lanes, num_cores)
          # exactness on the rank window: piece[r] == x_global[tis[rows_c[r]]] for r in [cs, ce)
          want = x_global[tis[rows_c[cs:ce]]]
          ok_comp &= (not oob) and (not np.isnan(piece[cs:ce]).any()) and np.array_equal(piece[cs:ce], want)
          pieces[c * rpc : (c + 1) * rpc] = piece
          # CONTRAST: raw buffer-position bounds on the chunk-sized call must fail somewhere
          _, oob_b = _emulate_sc_gather(xg_c, src_c, lo, hi, lanes, num_cores)
          wrong_rows = not (cs == lo and ce == hi)  # right rows only if positions == ranks
          if oob_b or wrong_rows:
            bad_buggy += 1
        # PLACEMENT: full-length call, bounds AS-IS; verify only WRITTEN piece rows are read
        buf, oob_p = _emulate_sc_gather(pieces, piece_pos, lo, hi, lanes, num_cores)
        seg = buf[lo:hi]
        want = x_global[tis[lo:hi]]
        ok_place &= (not oob_p) and (not np.isnan(seg).any()) and np.array_equal(seg, want)
      print(
          f"  route-B emu EP={ep_size} N={n_chunks}: compaction exact+in-bounds={ok_comp}, "
          f"placement exact+writes-read-only={ok_place}, buggy-bounds bad shard-chunks={bad_buggy}/{total}"
      )
      if not ok_comp:
        failures.append(f"route-B emu EP={ep_size} N={n_chunks}: compaction wrong/OOB")
      if not ok_place:
        failures.append(f"route-B emu EP={ep_size} N={n_chunks}: placement wrong or read unwritten piece rows")
      if n_chunks > 1 and bad_buggy == 0:
        failures.append(f"route-B emu EP={ep_size} N={n_chunks}: buffer-position bounds unexpectedly OK")


def main():
  devices = np.array(jax.devices())
  assert len(devices) >= 8, f"need 8 CPU devices, got {len(devices)}"
  failures = []

  print("SC dispatch addressing-invariant emulation (chunk_local_row in-bounds for xg_c):")
  _addressing_invariant_check(failures)

  print("SC Route-B addressing emulation (compaction rank-space bounds + placement):")
  _route_b_emulation_check(failures)

  for ep_size in (8, 4):
    mesh = jax.sharding.Mesh(devices[:ep_size], ("ep",))
    ntg = NUM_TOKENS_LOCAL * ep_size
    for case, skew in (("non-uniform", 3.0), ("uniform-ish", 0.0)):
      key = jax.random.PRNGKey(11)
      # x_local sharded over ep on dim0: global [ntg, hidden]
      x_local = jax.random.normal(key, (ntg, HIDDEN), dtype=jnp.float32) * 0.1
      topk_global = _make_topk(jax.random.fold_in(key, 2), ntg, skew)
      gs = np.asarray(jax.nn.one_hot(topk_global.reshape(-1), NUM_EXPERTS).sum(0))
      print(f"EP={ep_size} {case}: group_sizes std={gs.std():.1f} min={gs.min():.0f} max={gs.max():.0f}")

      ref = _unchunked(mesh, ep_size, x_local, topk_global)
      for n in (1, 2, 4, 8):
        out = _chunked(mesh, ep_size, x_local, topk_global, n)
        d = np.abs(np.asarray(ref) - np.asarray(out)).max()
        print(f"  dispatch N={n}: max|diff|={d:.3e}")
        if d > 0.0:
          failures.append(f"EP={ep_size} {case} fwd N={n} not bit-exact: max|diff|={d:.3e}")

      # ---- backward: grad wrt local x, chunked vs unchunked ----
      ct = jax.random.normal(jax.random.PRNGKey(5), (ep_size * ntg * TOPK, HIDDEN), dtype=jnp.float32)

      def loss_un(xl):
        def b(xl_, tg):
          xg = jax.lax.all_gather(xl_, "ep", axis=0, tiled=True)
          buf, _, _ = ring_ragged_sort(xg, tg, NUM_EXPERTS, TOPK, "ep", ep_size, buffer_size=None, **FB)
          return buf
        out = jax.shard_map(b, mesh=mesh, in_specs=(jax.P("ep"), jax.P()), out_specs=jax.P("ep"))(xl, topk_global)
        return jnp.vdot(out, ct)

      def loss_ch(xl, _n):
        def b(xl_, tg):
          buf, _, _ = chunked_ring_dispatch(xl_, tg, NUM_EXPERTS, TOPK, "ep", ep_size, _n, **FB)
          return buf
        out = jax.shard_map(b, mesh=mesh, in_specs=(jax.P("ep"), jax.P()), out_specs=jax.P("ep"))(xl, topk_global)
        return jnp.vdot(out, ct)

      g_ref = np.asarray(jax.jit(jax.grad(loss_un))(x_local))
      for n in (1, 2, 4, 8):
        g_ch = np.asarray(jax.jit(jax.grad(lambda xl, _n=n: loss_ch(xl, _n)))(x_local))
        gd = np.abs(g_ref - g_ch).max()
        print(f"  dispatch N={n} BWD: max|grad diff|={gd:.3e}")
        if gd > 0.0:
          failures.append(f"EP={ep_size} {case} bwd N={n} not bit-exact: max|grad diff|={gd:.3e}")

  print()
  if failures:
    print("FAILURES:")
    for f in failures:
      print(f"  {f}")
    raise SystemExit(1)
  print("ALL CASES PASSED (chunked dispatch == un-chunked reference)")


if __name__ == "__main__":
  main()
