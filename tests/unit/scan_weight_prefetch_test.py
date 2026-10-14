# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`scan_weight_prefetch=block_tie` against the scanned Qwen3.5 block, on a CPU mesh.

The switch gathers the selected dense kernels explicitly at the top of `Qwen3_5ScannableBlock` and ties them to the
input of one sub-layer with a forward-only optimization barrier. The barrier is the identity and the gathered bytes
are the ones the partitioner gathers anyway, so the loss and every gradient must be bit-identical with the switch
on and off.

Runs as a subprocess so the forced device count takes effect before JAX initializes. The mesh is 8 CPU devices,
fsdp2 x context2 x expert2 under cp-as-ep, so every dense kernel is sharded on embed over all three axes and
gathered per layer, as in the Qwen3.5-397B configuration (fsdp32 x context4 x expert2). The model is a shrunk
Qwen3.5-397B (two scan iterations of the 4-sub-layer cycle), `remat_policy=custom` with the tensors the 5409 recipe
keeps on device that are valid without the GDN kernel (`decoder_layer_input`, `context`, `moe_route`; the GDN
tensors stay rematerialized).

Checks:
  - knob off vs `block_tie` for both kernel sets: loss and every parameter gradient `jnp.array_equal`;
  - structure on the jaxpr (the CPU compiler strips optimization barriers): off emits no barrier, on emits one tie
    whose dependency operands are the gathered kernels (all-gather constraints to specs without the fsdp, context
    and expert axes), with a different tie sub-layer giving the same numerics;
  - the validator: default off, `scan_carry` reserved, scan_layers / decoder_block / tie-index requirements, and
    the knob switched off with a warning when --xla_tpu_aggressive_opt_barrier_removal=true or ENABLED applies
    (compile_xla_flags overriding LIBTPU_INIT_ARGS).
"""

import os
import subprocess
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common import common_types as ctypes
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.models import qwen3_5
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils.sharding import logical_to_mesh_sharding
from tests.utils.test_helpers import get_test_config_path

_NUM_DEVICES = 8
_SEQ = 32
_EMB = 256
_PASS_TOKEN = "SCAN_WEIGHT_PREFETCH_OK"


@pytest.mark.cpu_only
def test_block_tie_in_subprocess():
  """Runs the numeric and structural comparison in a child process with eight CPU devices."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={_NUM_DEVICES}"
  env["JAX_PLATFORMS"] = "cpu"
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  env["PYTHONPATH"] = os.pathsep.join([repo_root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else repo_root
  result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _cfg(prefetch, **overrides):
  """The shrunk scanned Qwen3.5-397B on an fsdp2 x context2 x expert2 cp-as-ep mesh."""
  kwargs = {
      "run_name": f"scan_weight_prefetch_{prefetch}",
      "base_output_directory": tempfile.mkdtemp(prefix="scan_weight_prefetch_"),
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
      "model_name": "qwen3.5-397b-a17b",
      "override_model_config": True,
      "base_num_decoder_layers": 8,
      "base_emb_dim": _EMB,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 256,
      "base_mlp_dim": 128,
      "base_moe_mlp_dim": 128,
      "num_experts": 8,
      "num_experts_per_tok": 2,
      "shared_experts": 1,
      "gdn_key_head_dim": 64,
      "gdn_value_head_dim": 64,
      "gdn_num_key_heads": 4,
      "gdn_num_value_heads": 8,
      "vocab_size": 1024,
      "max_target_length": _SEQ,
      "max_prefill_predict_length": _SEQ,
      "per_device_batch_size": 0.5,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "attention": "dot_product",
      # The ring-of-experts ragged-sort MoE path of the recipe (`moe_route=device` needs it); the GMM is megablox in
      # interpret mode, since tokamax gmm v2 does not run on CPU (as in moe_expert_weight_prefetch_test.py).
      "megablox": True,
      "sparse_matmul": True,
      "use_tokamax_gmm": False,
      "use_gmm_v2": False,
      "use_ring_of_experts": True,
      "use_ragged_sort": True,
      "use_custom_sort_vjp": False,
      "ragged_buffer_factor": 2.0,
      "num_moe_token_chunks": 2,
      "ring_of_experts_local_routing": True,
      "scan_layers": True,
      "remat_policy": "custom",
      "decoder_layer_input": "device",
      "context": "device",
      "moe_route": "device",
      "custom_mesh_and_rule": "cp-as-ep",
      "ici_fsdp_parallelism": 2,
      "ici_context_parallelism": 2,
      "ici_expert_parallelism": 2,
      "context_parallel_load_balance": False,
      "scan_weight_prefetch": prefetch,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


def _batch(cfg):
  batch = int(cfg.global_batch_size_to_train_on)
  rng = np.random.default_rng(0)
  inputs = jnp.asarray(rng.integers(1, cfg.vocab_size, size=(batch, _SEQ)), jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(_SEQ, dtype=jnp.int32)[None, :], (batch, _SEQ))
  segmentation = jnp.ones((batch, _SEQ), jnp.int32)
  return {
      "inputs": inputs,
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets": jnp.roll(inputs, -1, axis=-1),
      "targets_segmentation": segmentation,
  }


def _subjaxprs(eqn):
  for param in eqn.params.values():
    for sub in param if isinstance(param, (list, tuple)) else (param,):
      sub = sub if hasattr(sub, "eqns") else getattr(sub, "jaxpr", None)
      if sub is not None and hasattr(sub, "eqns"):
        yield sub


def _walk(jaxpr):
  for eqn in jaxpr.eqns:
    yield eqn
    for sub in _subjaxprs(eqn):
      yield from _walk(sub)


def _barriers(jaxpr):
  """The tie equations: a bare optimization_barrier, or the custom_vjp call whose body is one (the forward-only
  barrier of `barrier_tie`), so that the operands are variables of the enclosing jaxpr."""
  ties = []

  def walk(j):
    for eqn in j.eqns:
      subs = list(_subjaxprs(eqn))
      if eqn.primitive.name == "optimization_barrier":
        ties.append(eqn)
      elif eqn.primitive.name.startswith("custom_vjp") and any(
          e.primitive.name == "optimization_barrier" for s in subs for e in s.eqns
      ):
        ties.append(eqn)
      else:
        for s in subs:
          walk(s)

  walk(jaxpr)
  return ties


def _gathered_dependency_count(jaxpr, barrier):
  """How many of the barrier's dependency operands come from sharding constraints without fsdp/context/expert."""
  producers = {}
  for eqn in _walk(jaxpr):
    for out in eqn.outvars:
      producers[id(out)] = eqn
  n_gathered = 0
  for var in barrier.invars[1:]:  # invars[0] is the tied activation
    eqn = producers.get(id(var))
    while eqn is not None and eqn.primitive.name in ("convert_element_type", "pjit", "closed_call"):
      eqn = producers.get(id(eqn.invars[0]))
    if eqn is None or eqn.primitive.name != "sharding_constraint":
      continue
    spec = eqn.params["sharding"].spec
    names = {a for axis in spec if axis is not None for a in (axis if isinstance(axis, tuple) else (axis,))}
    if not names & {"fsdp", "fsdp_transpose", "context", "expert"}:
      n_gathered += 1
  return n_gathered


def _run(cfg, mesh, params_template=None):
  """Loss and parameter gradients of `train.loss_fn` for `cfg`; returns (loss, grads, params, barrier info)."""
  rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=jax.random.PRNGKey(42))
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=ctypes.MODEL_MODE_TRAIN)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    if params_template is not None:
      params = params_template
    else:
      shardings = logical_to_mesh_sharding(nnx.get_partition_spec(params), mesh, cfg.logical_axis_rules)
      params = jax.device_put(params, shardings)
    batch = _batch(cfg)

    def loss_fn(p):
      m = nnx.merge(graphdef, p, rest)
      loss, _ = train.loss_fn(m, cfg, batch, dropout_rng=None, params=None, is_train=True)
      return loss

    jaxpr = jax.make_jaxpr(loss_fn)(params).jaxpr
    barriers = _barriers(jaxpr)
    gathered = [_gathered_dependency_count(jaxpr, b) for b in barriers]
    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)
    jax.block_until_ready((loss, grads))
  return loss, grads, params, (len(barriers), [len(b.invars) - 1 for b in barriers], gathered)


def _compare(label, ref, new):
  loss0, grads0 = ref
  loss1, grads1 = new
  ok = bool(jnp.array_equal(loss0, loss1))
  leaves0 = jax.tree_util.tree_leaves_with_path(grads0)
  leaves1 = jax.tree_util.tree_leaves(grads1)
  n_diff = 0
  for (path, a), b in zip(leaves0, leaves1):
    if not bool(jnp.array_equal(a, b)):
      n_diff += 1
      if n_diff <= 5:
        print(f"{label}: gradient differs at {jax.tree_util.keystr(path)}", flush=True)
  ok &= n_diff == 0
  print(
      f"{label}: [{'ok' if ok else 'FAIL'}] loss {float(loss0):.6f} vs {float(loss1):.6f} bit-equal="
      f"{bool(jnp.array_equal(loss0, loss1))}; {len(leaves1)} gradient leaves, {n_diff} differ",
      flush=True,
  )
  return ok


def main():
  assert jax.device_count() == _NUM_DEVICES, jax.device_count()
  cfg_off = _cfg("none")
  assert cfg_off.scan_weight_prefetch is None and cfg_off.moe_expert_weight_prefetch is None
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg_off), cfg_off.mesh_axes)
  print("mesh", dict(mesh.shape), "layers", cfg_off.num_decoder_layers, "cycle", cfg_off.inhomogeneous_layer_cycle_interval)
  loss0, grads0, params, (n_bar, _, _) = _run(cfg_off, mesh)
  print(f"off: barriers={n_bar} loss={float(loss0):.6f}", flush=True)
  ok = n_bar == 0

  cases = (  # label, set, tie sub-layer, expected number of tied kernels
      ("block_tie attn_q_shared_wi tie=2", "attn_q_shared_wi", 2, 1 + 2 * 2),  # q + wi_0/wi_1 of sub-layers 2, 3
      ("block_tie attn_q_shared_wi tie=1", "attn_q_shared_wi", 1, 1 + 3 * 2),
      ("block_tie attn_layer tie=2", "attn_layer", 2, 4 + 3 + 1),  # q, k, v, out + wi_0, wi_1, wo + gate
      ("block_tie attn_layer tie=3", "attn_layer", 3, 4 + 3 + 1),
  )
  for label, kernel_set, tie, n_expected in cases:
    cfg_on = _cfg("block_tie", scan_weight_prefetch_set=kernel_set, scan_weight_prefetch_tie_sublayer=tie)
    assert cfg_on.scan_weight_prefetch == "block_tie", cfg_on.scan_weight_prefetch
    loss1, grads1, _, (n_bar, n_deps, n_gathered) = _run(cfg_on, mesh, params_template=params)
    structural = n_bar >= 1 and all(d == n_expected for d in n_deps) and all(g == n_expected for g in n_gathered)
    print(
        f"{label}: [{'ok' if structural else 'FAIL'}] barriers={n_bar} tied operands={n_deps} gathered={n_gathered}"
        f" expected {n_expected}",
        flush=True,
    )
    ok &= structural
    ok &= _compare(label, (loss0, grads0), (loss1, grads1))

  assert ok, "see the FAIL lines above"
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()


class ScanWeightPrefetchConfigTest(absltest.TestCase):
  """The validator, on the plain test config (no devices needed)."""

  def _init(self, **kwargs):
    kwargs.setdefault("run_name", "scan_weight_prefetch_config")
    kwargs.setdefault("enable_checkpointing", False)
    kwargs.setdefault("skip_jax_distributed_system", True)
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)

  def _qwen(self, **kwargs):
    kwargs.setdefault("model_name", "qwen3.5-397b-a17b")
    kwargs.setdefault("override_model_config", True)
    kwargs.setdefault("base_num_decoder_layers", 4)
    kwargs.setdefault("scan_layers", True)
    return self._init(**kwargs)

  def test_defaults_to_off(self):
    cfg = self._init()
    self.assertIsNone(cfg.scan_weight_prefetch)
    self.assertEqual(cfg.scan_weight_prefetch_set, "attn_q_shared_wi")
    self.assertEqual(cfg.scan_weight_prefetch_tie_sublayer, 2)
    self.assertEqual(cfg.scan_weight_prefetch_sparse_core_id, -1)

  def test_block_tie_on_scanned_qwen3_5(self):
    cfg = self._qwen(scan_weight_prefetch="block_tie")
    self.assertEqual(cfg.scan_weight_prefetch, "block_tie")
    self.assertEqual(cfg.decoder_block, ctypes.DecoderBlockType.QWEN3_5)

  def test_scan_carry_is_reserved(self):
    with self.assertRaises(NotImplementedError):
      self._qwen(scan_weight_prefetch="scan_carry")

  def test_rejects_unknown_mode(self):
    with self.assertRaisesRegex(ValueError, "scan_weight_prefetch"):
      self._qwen(scan_weight_prefetch="early")

  def test_requires_scan_layers(self):
    with self.assertRaisesRegex(ValueError, "scan_layers"):
      self._qwen(scan_weight_prefetch="block_tie", scan_layers=False)

  def test_requires_qwen3_5_block(self):
    with self.assertRaisesRegex(ValueError, "decoder_block=qwen3_5"):
      self._init(scan_weight_prefetch="block_tie", scan_layers=True)

  def test_tie_sublayer_in_cycle(self):
    with self.assertRaisesRegex(ValueError, "scan_weight_prefetch_tie_sublayer"):
      self._qwen(scan_weight_prefetch="block_tie", scan_weight_prefetch_tie_sublayer=4)
    with self.assertRaisesRegex(ValueError, "scan_weight_prefetch_tie_sublayer"):
      self._qwen(scan_weight_prefetch="block_tie", scan_weight_prefetch_tie_sublayer=-1)

  def test_disabled_when_barriers_are_stripped(self):
    flag = "--xla_tpu_aggressive_opt_barrier_removal"
    cases = (  # label, compile_xla_flags, LIBTPU_INIT_ARGS, whether the knob stays on
        ("flag unset", "", "", True),
        ("compile_xla_flags true", f"{flag}=true", "", False),
        ("LIBTPU_INIT_ARGS true", "", f"{flag}=true", False),
        ("LIBTPU_INIT_ARGS ENABLED", "", f"{flag}=ENABLED", False),
        ("LIBTPU_INIT_ARGS true, compile_xla_flags false", f"{flag}=false", f"{flag}=true", True),
        ("LIBTPU_INIT_ARGS false, compile_xla_flags true", f"{flag}=true", f"{flag}=false", False),
    )
    for label, compile_flags, env_flags, kept in cases:
      with self.subTest(label), mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": env_flags}), mock.patch.object(
          config_types.max_logging, "warning"
      ) as warning:
        cfg = self._qwen(scan_weight_prefetch="block_tie", compile_xla_flags=compile_flags)
        warned = any("scan_weight_prefetch" in str(call) for call in warning.call_args_list)
        self.assertEqual(cfg.scan_weight_prefetch, "block_tie" if kept else None, label)
        self.assertEqual(warned, not kept, label)

  def test_module_selection_sets(self):
    """The kernel sets the block gathers, on an unscanned tiny block (no devices needed)."""
    cfg = self._qwen(
        base_emb_dim=64,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=256,
        base_mlp_dim=32,
        base_moe_mlp_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        shared_experts=1,
        gdn_key_head_dim=16,
        gdn_value_head_dim=16,
        gdn_num_key_heads=2,
        gdn_num_value_heads=4,
        vocab_size=64,
        max_target_length=8,
        max_prefill_predict_length=8,
        per_device_batch_size=1.0,
        attention="dot_product",
        scan_weight_prefetch="block_tie",
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      block = qwen3_5.Qwen3_5ScannableBlock(cfg, mesh, ctypes.MODEL_MODE_TRAIN, rngs=nnx.Rngs(params=0))
    paths = [p for _, p, _ in block._scan_weight_prefetch_modules()]  # pylint: disable=protected-access
    self.assertEqual(
        paths,
        ["layer_3.attention.attention.query"]
        + [f"layer_{i}.mlp.shared_expert.wi_{k}" for i in range(4) for k in range(2)],
    )
    object.__setattr__(cfg, "scan_weight_prefetch_set", "attn_layer")
    paths = [p for _, p, _ in block._scan_weight_prefetch_modules()]  # pylint: disable=protected-access
    self.assertEqual(
        paths,
        [
            "layer_3.attention.attention.query",
            "layer_3.attention.attention.key",
            "layer_3.attention.attention.value",
            "layer_3.attention.attention.out",
            "layer_3.mlp.shared_expert.wi_0",
            "layer_3.mlp.shared_expert.wi_1",
            "layer_3.mlp.shared_expert.wo",
            "layer_3.mlp.routed_experts.gate",
        ],
    )
