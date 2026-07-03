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
token axis; each chunk's tokens scatter across the WHOLE expert-sorted buffer, so this exercises
the disjoint-write buffer fill (fwd) and the un-chunked ragged_gather_reduce + psum_scatter (bwd).
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


def main():
  devices = np.array(jax.devices())
  assert len(devices) >= 8, f"need 8 CPU devices, got {len(devices)}"
  failures = []

  print("SC dispatch addressing-invariant emulation (chunk_local_row in-bounds for xg_c):")
  _addressing_invariant_check(failures)

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
        if d > 1e-6:
          failures.append(f"EP={ep_size} {case} fwd N={n}: max|diff|={d:.3e}")

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
        if gd > 1e-6:
          failures.append(f"EP={ep_size} {case} bwd N={n}: max|grad diff|={gd:.3e}")

  print()
  if failures:
    print("FAILURES:")
    for f in failures:
      print(f"  {f}")
    raise SystemExit(1)
  print("ALL CASES PASSED (chunked dispatch == un-chunked reference)")


if __name__ == "__main__":
  main()
