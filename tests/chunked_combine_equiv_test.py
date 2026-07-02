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

"""CPU equivalence test: chunked combine+RS vs un-chunked ring_ragged_unsort + psum_scatter.

Run with:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
      python3 tests/chunked_combine_equiv_test.py

Covers the failure mode that the old EP=4/uniform test missed: EP=8 with NON-UNIFORM
random group sizes across experts/shards, and GARBAGE values in the buffer positions
OUTSIDE each shard's valid [shard_output_start, shard_output_end) range (in the real
model those rows are uninitialized GMM output -- any mask/shard misalignment gathers
them and blows up the output, which is exactly the observed step-0 NaN signature).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"

import functools

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.ragged.ragged_sort import chunked_ring_combine_reduce_scatter, ring_ragged_unsort

TOPK = 4
NUM_EXPERTS = 16
HIDDEN = 16
NUM_TOKENS = 128  # divisible by n_chunks * ep_size for all tested combos


def _make_routing(key, skew=3.0):
  """Random top-k routing with a strong per-expert bias -> NON-UNIFORM group sizes."""
  k_logits, k_bias = jax.random.split(key)
  logits = jax.random.normal(k_logits, (NUM_TOKENS, NUM_EXPERTS))
  logits = logits + skew * jax.random.normal(k_bias, (NUM_EXPERTS,))  # skew expert popularity
  topk_indices = jax.lax.top_k(logits, TOPK)[1].astype(jnp.int32)  # [T, topk], distinct per token
  flat = topk_indices.reshape(-1)
  sort_idx = jnp.argsort(flat)
  revert = jnp.argsort(sort_idx).astype(jnp.int32)  # topk_argsort_revert_indices
  group_sizes = jnp.sum(jax.nn.one_hot(flat, NUM_EXPERTS, dtype=jnp.int32), axis=0)
  return revert, group_sizes


def _shard_ranges(group_sizes, ep_size):
  offsets = np.concatenate([[0], np.cumsum(np.asarray(group_sizes))])
  local_e = NUM_EXPERTS // ep_size
  starts = offsets[np.arange(ep_size) * local_e]
  ends = offsets[(np.arange(ep_size) + 1) * local_e]
  return starts, ends


def _make_full_buffers(key, group_sizes, ep_size):
  """Per-shard FULL (mode-1) buffers: true values inside [start, end), garbage outside."""
  n_slots = NUM_TOKENS * TOPK
  k_true, k_junk = jax.random.split(key)
  buf_true = jax.random.normal(k_true, (n_slots, HIDDEN), dtype=jnp.float32) * 1e-3
  starts, ends = _shard_ranges(group_sizes, ep_size)
  bufs = []
  for s in range(ep_size):
    pos = np.arange(n_slots)
    valid = (pos >= starts[s]) & (pos < ends[s])
    garbage = 30.0 * (s + 1) + jax.random.normal(jax.random.fold_in(k_junk, s), (n_slots, HIDDEN))
    bufs.append(jnp.where(jnp.asarray(valid)[:, None], buf_true, garbage))
  return jnp.stack(bufs), buf_true  # [ep, n_slots, H]


def _make_truncated_buffers(key, group_sizes, ep_size, buffer_size):
  """Per-shard PACKED (mode-2) buffers of size buffer_size < num_tokens*topk."""
  n_slots = NUM_TOKENS * TOPK
  k_true, k_junk = jax.random.split(key)
  buf_true = jax.random.normal(k_true, (n_slots, HIDDEN), dtype=jnp.float32) * 1e-3
  starts, ends = _shard_ranges(group_sizes, ep_size)
  bufs = []
  for s in range(ep_size):
    limit = min(int(ends[s] - starts[s]), buffer_size)
    src = np.clip(starts[s] + np.arange(buffer_size), 0, n_slots - 1)
    packed = jnp.asarray(np.asarray(buf_true)[src])
    valid = np.arange(buffer_size) < limit
    garbage = 30.0 * (s + 1) + jax.random.normal(jax.random.fold_in(k_junk, s), (buffer_size, HIDDEN))
    bufs.append(jnp.where(jnp.asarray(valid)[:, None], packed, garbage))
  return jnp.stack(bufs)  # [ep, buffer_size, H]


def _run(mesh, ep_size, bufs, group_sizes, revert, w_flat, n_chunks):
  """n_chunks == 0 -> un-chunked reference (ring_ragged_unsort + single psum_scatter)."""
  local_e = NUM_EXPERTS // ep_size

  # On CPU there is no SparseCore: force the pure-JAX fallback path explicitly
  # (pltpu.get_tpu_info() raises on non-TPU backends instead of returning sc_info=None).
  fb = dict(enforce_gather_fallback=True, enforce_gather_reduce_fallback=True)

  def body(buf, gs, ridx, wf):
    buf = buf[0]
    if n_chunks == 0:
      out = ring_ragged_unsort(buf, gs, ridx, TOPK, local_e, "ep", topk_weights=wf, **fb)
      return jax.lax.psum_scatter(out, "ep", scatter_dimension=0, tiled=True)
    return chunked_ring_combine_reduce_scatter(buf, gs, ridx, TOPK, local_e, "ep", wf, ep_size, n_chunks, **fb)

  fn = jax.shard_map(
      body,
      mesh=mesh,
      in_specs=(jax.P("ep"), jax.P(), jax.P(), jax.P()),
      out_specs=jax.P("ep"),
  )
  return jax.jit(fn)(bufs, group_sizes, revert, w_flat)  # [T, H] global (concat of shard slices)


def _report(name, ref, out, ep_size, n_chunks):
  diff = np.abs(np.asarray(ref) - np.asarray(out))
  per = NUM_TOKENS // (max(n_chunks, 1) * ep_size)
  # rows belonging to chunk 0 = first `per` rows of every shard's T/ep block
  row_in_shard = np.arange(NUM_TOKENS) % (NUM_TOKENS // ep_size)
  c0_mask = row_in_shard < per
  max_all = diff.max()
  max_c0 = diff[c0_mask].max()
  max_rest = diff[~c0_mask].max() if (~c0_mask).any() else 0.0
  print(f"  {name}: max|diff|={max_all:.3e}  (chunk0-rows {max_c0:.3e}, c>0-rows {max_rest:.3e})")
  return max_all, max_c0, max_rest


def _emulate_sc_gather_reduce(x, indices, weights, valid_mask, k, num_row_partitions, lanes, stride_from_buffer):
  """Numpy emulation of the SC ragged_gather_reduce wrapper + main_kernel ADDRESSING.

  Mirrors: wrapper padding, _preprocess (per-partition validity compaction, src/dst/weights),
  and inner_kernel's row-partition addressing (row_start = p * row_partition_size, reading
  ceil(nvalid_p / lanes) * lanes rows of the slot arrays). ``stride_from_buffer=True``
  reproduces the ORIGINAL bug: row_partition_size = in_hbm_ref.shape[0] // P (the x buffer's
  row count); ``False`` uses the FIXED expression: src_indices_hbm_ref.shape[0] // P. Raises
  IndexError when a partition would read the slot arrays out of bounds (on hardware this is a
  SILENT garbage read: disable_bounds_checks=True); otherwise returns the numeric result.
  """
  import math as _math

  n = indices.shape[0]
  p_cnt, hidden = num_row_partitions, x.shape[1]
  align = _math.lcm(p_cnt * lanes, k)
  padded = -(-n // align) * align
  idx = np.pad(indices, (0, padded - n))
  wts = np.pad(weights, (0, padded - n))
  msk = np.pad(valid_mask, (0, padded - n))
  x_rows = max(padded, x.shape[0])  # wrapper pads x rows up to padded_input_size if shorter
  xp = np.zeros((x_rows, hidden), np.float32)
  xp[: x.shape[0]] = x

  # _preprocess: compact valid slots to the front of each partition (stable), build src/dst/w
  rps = padded // p_cnt
  m2 = msk.reshape(p_cnt, rps)
  order = np.argsort(~m2, axis=-1, kind="stable") + np.arange(p_cnt)[:, None] * rps
  order = order.reshape(-1)
  src, dst, w_s = idx[order], order // k, wts[order]
  nvalid = m2.sum(axis=-1)

  # inner_kernel addressing
  stride = (x_rows if stride_from_buffer else padded) // p_cnt
  out = np.zeros((padded // k, hidden), np.float32)
  for p in range(p_cnt):
    n_rows = int(-(-int(nvalid[p]) // lanes) * lanes)  # kernel reads whole lane tiles
    row_start = p * stride
    if n_rows and row_start + n_rows > padded:
      raise IndexError(
          f"partition {p}: reads slot rows [{row_start}, {row_start + n_rows}) beyond "
          f"slot-array length {padded} (buffer rows {x_rows}); silent garbage on SC"
      )
    for j in range(int(nvalid[p])):
      r = row_start + j
      out[dst[r]] += w_s[r] * xp[src[r]]
  group_mask = msk.reshape(-1, k).any(axis=-1)
  out = np.where(group_mask[:, None], out, 0.0)
  return out[: n // k]


def sc_kernel_contract_checks(failures):
  """Emulated-SC checks: the buggy buffer-derived stride violates the slot-array bounds for
  EVERY n_chunks > 1 (N-independent, matching the cluster step-0 NaN); the fixed stride is
  exact for all N and identical to the old behavior at N=1."""
  lanes, p_cnt = 16, 2  # v7x-like: sc num_lanes=16; 16 subcores -> 8 col partitions x 2 row partitions
  key = jax.random.PRNGKey(0)
  revert, group_sizes = _make_routing(key, skew=3.0)
  n_slots = NUM_TOKENS * TOPK
  x = np.asarray(jax.random.normal(jax.random.fold_in(key, 1), (n_slots, HIDDEN), dtype=jnp.float32))
  w = np.asarray(jax.random.uniform(jax.random.fold_in(key, 2), (n_slots,), dtype=jnp.float32))
  starts, ends = _shard_ranges(group_sizes, 8)
  rv = np.asarray(revert)
  s = 3  # emulate shard 3's combine call
  mask_full = (rv >= starts[s]) & (rv < ends[s])

  for n_chunks in (1, 2, 4, 8):
    spc = n_slots // n_chunks
    ok_fixed = True
    oob_chunks = 0
    for c in range(n_chunks):
      sl = slice(c * spc, (c + 1) * spc)
      ref = (x[rv[sl]] * w[sl][:, None] * mask_full[sl][:, None]).reshape(-1, TOPK, HIDDEN).sum(axis=1)
      got = _emulate_sc_gather_reduce(x, rv[sl], w[sl], mask_full[sl], TOPK, p_cnt, lanes, False)
      ok_fixed &= np.allclose(ref, got, atol=1e-5)
      try:
        _emulate_sc_gather_reduce(x, rv[sl], w[sl], mask_full[sl], TOPK, p_cnt, lanes, True)
      except IndexError:
        oob_chunks += 1
    print(f"  SC-emulation N={n_chunks}: fixed-stride exact={ok_fixed}, buggy-stride OOB chunks={oob_chunks}/{n_chunks}")
    if not ok_fixed:
      failures.append(f"SC-emulation N={n_chunks}: fixed stride numerically wrong")
    if n_chunks > 1 and oob_chunks == 0:
      failures.append(f"SC-emulation N={n_chunks}: buggy stride did NOT violate bounds (expected OOB)")
    if n_chunks == 1 and oob_chunks:
      failures.append("SC-emulation N=1: un-chunked call must not go OOB with either stride")


def main():
  devices = np.array(jax.devices())
  assert len(devices) >= 8, f"need 8 CPU devices, got {len(devices)}"
  failures = []

  print("SC ragged_gather_reduce kernel-contract emulation (row-partition stride):")
  sc_kernel_contract_checks(failures)

  for ep_size in (8, 4):
    mesh = jax.sharding.Mesh(devices[:ep_size], ("ep",))
    for case, skew in (("non-uniform", 3.0), ("uniform-ish", 0.0)):
      key = jax.random.PRNGKey(42)
      k_route, k_buf, k_w = jax.random.split(key, 3)
      revert, group_sizes = _make_routing(k_route, skew=skew)
      gs_np = np.asarray(group_sizes)
      w_flat = jax.random.uniform(k_w, (NUM_TOKENS * TOPK,), dtype=jnp.float32)

      # ---- full-buffer (mode 1) ----
      bufs, buf_true = _make_full_buffers(k_buf, group_sizes, ep_size)
      ref = _run(mesh, ep_size, bufs, group_sizes, revert, w_flat, 0)

      # dense ground truth: out[t] = sum_k w[t*topk+k] * buf_true[revert[t*topk+k]]
      gt = (np.asarray(buf_true)[np.asarray(revert)] * np.asarray(w_flat)[:, None]).reshape(
          NUM_TOKENS, TOPK, HIDDEN
      ).sum(axis=1)
      gt_err = np.abs(gt - np.asarray(ref)).max()
      print(f"EP={ep_size} {case}: group_sizes std={gs_np.std():.1f} min={gs_np.min()} max={gs_np.max()}; "
            f"unchunked-ref vs dense ground truth max|diff|={gt_err:.3e}")
      if gt_err > 2e-6:
        failures.append(f"EP={ep_size} {case}: reference itself wrong ({gt_err:.3e})")

      for n in (1, 2, 4, 8):
        out = _run(mesh, ep_size, bufs, group_sizes, revert, w_flat, n)
        max_all, _, _ = _report(f"full-buffer  N={n}", ref, out, ep_size, n)
        if max_all > 1e-6:
          failures.append(f"EP={ep_size} {case} full-buffer N={n}: max|diff|={max_all:.3e}")

      # ---- autodiff backward (grad wrt the expert-sorted buffer), full-buffer mode ----
      # Rung 6 exercises the chunked path under plain autodiff too (manbwd recomputes with the
      # un-chunked combine, but the flag must be safe without manbwd). The per-chunk
      # ring_ragged_unsort bwd emits an n-row grad scattered back to the buffer positions the
      # chunk's slice read; chunks are disjoint, so the summed grads must equal the un-chunked bwd.
      ct = jax.random.normal(jax.random.PRNGKey(7), (NUM_TOKENS, HIDDEN), dtype=jnp.float32)
      grads = {}
      for n in (0, 1, 2, 4, 8):
        loss_fn = lambda b, _n=n: jnp.vdot(_run(mesh, ep_size, b, group_sizes, revert, w_flat, _n), ct)
        grads[n] = np.asarray(jax.grad(loss_fn)(bufs))
      for n in (1, 2, 4, 8):
        gdiff = np.abs(grads[n] - grads[0]).max()
        print(f"  full-buffer  N={n} BWD: max|grad diff|={gdiff:.3e}")
        if gdiff > 1e-6:
          failures.append(f"EP={ep_size} {case} full-buffer N={n} BWD: max|grad diff|={gdiff:.3e}")

      # ---- truncated packed buffer (mode 2 un-chunked; exercises the mode boundary) ----
      buffer_size = 96 if ep_size == 8 else 192  # < T*topk/N for small N, >= for large N
      bufs_t = _make_truncated_buffers(k_buf, group_sizes, ep_size, buffer_size)
      ref_t = _run(mesh, ep_size, bufs_t, group_sizes, revert, w_flat, 0)
      for n in (1, 2, 4, 8):
        out_t = _run(mesh, ep_size, bufs_t, group_sizes, revert, w_flat, n)
        max_all, _, _ = _report(f"trunc-buffer N={n} (B={buffer_size})", ref_t, out_t, ep_size, n)
        if max_all > 1e-6:
          failures.append(f"EP={ep_size} {case} trunc-buffer N={n}: max|diff|={max_all:.3e}")

  print()
  if failures:
    print("FAILURES:")
    for f in failures:
      print(f"  {f}")
    raise SystemExit(1)
  print("ALL CASES PASSED (chunked == un-chunked reference)")


if __name__ == "__main__":
  main()
