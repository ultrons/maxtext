# Copyright 2023–2026 Google LLC
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

"""Tests for defer_small_all_reduces: the stacked partial expert counts and overflow flags, reduced once after the
layer loop, equal the values the per-layer all-reduces give."""

import os
import subprocess
import sys
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from tests.utils.test_helpers import get_test_config_path

_REQUIRED_CPU_DEVICES = 8


def _cfg(global_counts, rate=0.01, defer=True, ga=1):
  return SimpleNamespace(
      gradient_accumulation_steps=ga,
      routed_bias_update_rate=rate,
      routed_bias_global_counts=global_counts,
      defer_small_all_reduces=defer,
  )


def _per_layer_reference(local_counts, cfg):
  """What the layer emits without deferral: psum of the counts per chunk, then the chunk combine.

  local_counts: (num_layers, num_parts, num_chunks, E) int32.
  """
  out = []
  for layer in range(local_counts.shape[0]):
    chunk_signals = []
    for c in range(local_counts.shape[2]):
      global_counts = jnp.sum(local_counts[layer, :, c, :], axis=0)  # the per-layer psum
      if moe.routed_bias_emits_expert_counts(cfg):
        chunk_signals.append(global_counts)
      else:
        chunk_signals.append(moe.expert_counts_to_bias_updates(global_counts, cfg.routed_bias_update_rate))
    out.append(chunk_signals[0] if len(chunk_signals) == 1 else moe.combine_chunk_bias_signals(chunk_signals, cfg))
  return jnp.stack(out)


class FinalizeDeferredBiasSignalTest(unittest.TestCase):
  """Pure array tests (any device count)."""

  def _local_counts(self, num_layers=3, num_parts=4, num_chunks=2, num_experts=8, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, 50, size=(num_layers, num_parts, num_chunks, num_experts)), jnp.int32)

  def test_default_mode_equals_per_layer_chunk_average(self):
    """routed_bias_global_counts off (the R1 setting): per-chunk update from the reduced counts, then chunk mean."""
    local = self._local_counts()
    cfg = _cfg(global_counts=False)
    got = moe.finalize_deferred_bias_signal(local, cfg)
    ref = _per_layer_reference(local, cfg)
    self.assertEqual(got.shape, (3, 8))
    self.assertEqual(got.dtype, ref.dtype)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

  def test_global_counts_mode_equals_summed_counts(self):
    local = self._local_counts(seed=1)
    cfg = _cfg(global_counts=True)
    got = moe.finalize_deferred_bias_signal(local, cfg)
    self.assertEqual(got.dtype, jnp.int32)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(_per_layer_reference(local, cfg)))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.sum(local, axis=(1, 2))))

  def test_single_chunk_and_unscanned_layer(self):
    """One token chunk, and the unscanned (MTP) layer shape (num_parts, num_chunks, E) without a layer axis."""
    for global_counts in (False, True):
      cfg = _cfg(global_counts=global_counts)
      local = self._local_counts(num_chunks=1, seed=2)
      np.testing.assert_array_equal(
          np.asarray(moe.finalize_deferred_bias_signal(local, cfg)), np.asarray(_per_layer_reference(local, cfg))
      )
      single = self._local_counts(num_layers=1, seed=3)
      np.testing.assert_array_equal(
          np.asarray(moe.finalize_deferred_bias_signal(single[0], cfg)),
          np.asarray(_per_layer_reference(single, cfg)[0]),
      )

  def test_finalize_intermediates_rewrites_only_bias_leaves(self):
    local = self._local_counts(seed=4)
    flags = jnp.array([[False, True], [False, False], [False, False]])
    tree = {
        "decoder": {"moe_layers": {"moe_bias_updates": (local,), "moe_has_overflow": (flags,)}},
        "mtp_block": {"mtp_layer_1": {"moe_bias_updates": (local[0],)}},
    }
    cfg = _cfg(global_counts=False)
    out = moe.finalize_deferred_intermediates(tree, cfg)
    np.testing.assert_array_equal(
        np.asarray(out["decoder"]["moe_layers"]["moe_bias_updates"][0]), np.asarray(_per_layer_reference(local, cfg))
    )
    np.testing.assert_array_equal(
        np.asarray(out["mtp_block"]["mtp_layer_1"]["moe_bias_updates"][0]),
        np.asarray(_per_layer_reference(local[:1], cfg)[0]),
    )
    self.assertIs(out["decoder"]["moe_layers"]["moe_has_overflow"][0], flags)
    # Flag off: the tree is returned as is.
    self.assertIs(moe.finalize_deferred_intermediates(tree, _cfg(global_counts=False, defer=False)), tree)


class DeferConfigTest(unittest.TestCase):
  """The flag's default and its validation."""

  def _init(self, **kw):
    return pyconfig.initialize(
        [None, get_test_config_path()], run_name="defer_small_ar_test", enable_checkpointing=False, **kw
    )

  def test_default_off(self):
    self.assertFalse(self._init().defer_small_all_reduces)

  def test_requires_ring_of_experts_sparse_matmul(self):
    with self.assertRaisesRegex(ValueError, "defer_small_all_reduces requires use_ring_of_experts"):
      self._init(defer_small_all_reduces=True, use_ring_of_experts=False, sparse_matmul=True)


@pytest.mark.cpu_only
def test_deferred_reduction_on_cpu_mesh():
  """Runs the mesh tests below in a subprocess with 8 forced CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_REQUIRED_CPU_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "DEFER_SMALL_AR_MESH_TESTS_PASSED" in result.stdout


def _count_prims(jaxpr, names, inside_scan=False):
  """Returns (#name eqns inside a scan body, #name eqns outside) over a closed jaxpr, recursively."""
  inside, outside = 0, 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in names:
      if inside_scan:
        inside += 1
      else:
        outside += 1
    for v in eqn.params.values():
      for sub in v if isinstance(v, (list, tuple)) else (v,):
        sub_jaxpr = getattr(sub, "jaxpr", sub)
        if hasattr(sub_jaxpr, "eqns"):
          i, o = _count_prims(sub_jaxpr, names, inside_scan or eqn.primitive.name == "scan")
          inside, outside = inside + i, outside + o
  return inside, outside


class DeferredMeshTest(unittest.TestCase):
  """A scanned stack of shard_map 'layers' on a (fsdp=4, expert=2) CPU mesh, shaped like the ring-of-experts path:
  tokens are sharded over fsdp and replicated over expert, counts are psum'd over fsdp only, the overflow flag over
  the whole mesh."""

  __test__ = False

  num_layers, num_chunks, num_experts, rate = 3, 2, 8, 0.01

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < _REQUIRED_CPU_DEVICES:
      self.skipTest("needs 8 CPU devices; run through test_deferred_reduction_on_cpu_mesh")
    self.mesh = Mesh(np.array(jax.devices("cpu")[:8]).reshape(4, 2), ("fsdp", "expert"))
    rng = np.random.default_rng(0)
    # (layers, tokens, top_k) expert ids, tokens sharded over fsdp in the shard_map.
    self.indices = jnp.asarray(rng.integers(0, self.num_experts, size=(self.num_layers, 64, 2)), jnp.int32)
    # Local overflow on exactly one device (fsdp=2, expert=1) in layer 1 only.
    ov = np.zeros((self.num_layers, 4, 2), np.int32)
    ov[1, 2, 1] = 1
    self.local_overflow = jnp.asarray(ov)

  def _run(self, cfg, defer, bias_out_spec=None, flag_out_spec=None):
    """Returns ((bias signal, any overflow, per-layer overflow), jaxpr) of the scanned stack, deferred or not."""
    mesh, n_chunks, e = self.mesh, self.num_chunks, self.num_experts

    def layer(idx, ov):
      chunks = jnp.split(idx, n_chunks, axis=0)
      fsdp_i, ep_i = jax.lax.axis_index("fsdp"), jax.lax.axis_index("expert")
      local_ov = ov[fsdp_i, ep_i]
      if defer:
        counts = jnp.stack([moe.calculate_expert_counts(c, e) for c in chunks])[None]  # (1, n_chunks, E)
        return counts, jnp.reshape(local_ov > 0, (1,))
      signals = [moe.calculate_routed_bias_signal(c, e, cfg, axis_names=("fsdp",)) for c in chunks]
      return moe.combine_chunk_bias_signals(signals, cfg), jax.lax.psum(local_ov, ("fsdp", "expert")) > 0

    bias_spec = bias_out_spec if bias_out_spec is not None else (P("fsdp", None, None) if defer else P())
    flag_spec = flag_out_spec if flag_out_spec is not None else (P(("fsdp", "expert")) if defer else P())
    smap = jax.shard_map(
        layer, mesh=mesh, in_specs=(P("fsdp", None), P()), out_specs=(bias_spec, flag_spec), check_vma=False
    )

    def model(indices, overflow):
      _, (bias, flags) = jax.lax.scan(lambda carry, xs: (carry, smap(*xs)), None, (indices, overflow))
      tree = (
          moe.finalize_deferred_intermediates({"moe_bias_updates": (bias,)}, cfg)
          if defer
          else {"moe_bias_updates": (bias,)}
      )
      return tree["moe_bias_updates"][0], jnp.any(flags), jnp.any(flags, axis=tuple(range(1, flags.ndim)))

    jaxpr = jax.make_jaxpr(model)(self.indices, self.local_overflow)
    return jax.jit(model)(self.indices, self.local_overflow), jaxpr

  def test_deferred_equals_per_layer(self):
    for global_counts in (False, True):
      cfg = _cfg(global_counts=global_counts, rate=self.rate)
      (ref_bias, ref_any, ref_per_layer), ref_jaxpr = self._run(cfg, defer=False)
      (bias, any_flag, per_layer), jaxpr = self._run(cfg, defer=True)
      np.testing.assert_array_equal(np.asarray(bias), np.asarray(ref_bias))
      self.assertTrue(bool(ref_any))
      self.assertEqual(bool(any_flag), bool(ref_any))
      np.testing.assert_array_equal(np.asarray(per_layer), np.asarray(ref_per_layer))
      np.testing.assert_array_equal(np.asarray(per_layer), np.array([False, True, False]))
      # The reference has one count psum per chunk and one flag psum per layer inside the scan; the deferred model has
      # none there.
      self.assertEqual(_count_prims(ref_jaxpr.jaxpr, ("psum", "psum2", "psum_invariant"))[0], self.num_chunks + 1)
      self.assertEqual(_count_prims(jaxpr.jaxpr, ("psum", "psum2", "psum_invariant"))[0], 0)

  def test_replicated_out_spec_is_wrong(self):
    """Negative control: emitting the unreduced partials under P() (the pre-deferral out_spec) loses the other
    devices' counts and flags, so the test above would catch that mistake."""
    cfg = _cfg(global_counts=True, rate=self.rate)
    (ref_bias, ref_any, _), _ = self._run(cfg, defer=False)
    (bias, any_flag, _), _ = self._run(cfg, defer=True, bias_out_spec=P(None, None, None), flag_out_spec=P(None))
    self.assertFalse(np.array_equal(np.asarray(bias), np.asarray(ref_bias)))
    self.assertNotEqual(bool(any_flag), bool(ref_any))


if __name__ == "__main__":
  DeferredMeshTest.__test__ = True
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeferredMeshTest)
  res = unittest.TextTestRunner(verbosity=2).run(suite)
  if res.wasSuccessful() and res.testsRun == 2 and not res.skipped:
    print("DEFER_SMALL_AR_MESH_TESTS_PASSED")
  sys.exit(0 if res.wasSuccessful() else 1)
