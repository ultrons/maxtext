# Copyright 2026 Google LLC
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

"""eval_sample_repeat: the eval step tiles the real eval rows k times and must report the k=1 loss and weights."""

import functools
import sys
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train as pre_train
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

_N_REAL = 2  # real eval samples in every test batch
_N_LOADED = 8  # rows the loader delivers; rows >= _N_REAL are junk that must never reach the loss
_SEQ = 16


def _loaded_batch(vocab_size, seq=_SEQ, seed=0):
  """A loaded eval batch: _N_REAL real rows (with padding tails) followed by junk rows with nonzero weight."""
  rng = np.random.default_rng(seed)
  inputs = rng.integers(1, vocab_size, size=(_N_LOADED, seq), dtype=np.int32)
  targets = np.roll(inputs, -1, axis=1)
  seg = np.ones((_N_LOADED, seq), dtype=np.int32)
  # Real rows: two packed documents in row 0, a padded tail in row 1, so weights differ per row.
  seg[0, seq // 2 :] = 2
  seg[1, seq - 5 :] = 0
  pos = np.tile(np.arange(seq, dtype=np.int32), (_N_LOADED, 1))
  pos[0, seq // 2 :] = np.arange(seq - seq // 2)
  targets = targets * (seg != 0)
  return {
      "inputs": jnp.asarray(inputs),
      "inputs_position": jnp.asarray(pos),
      "inputs_segmentation": jnp.asarray(seg),
      "targets": jnp.asarray(targets),
      "targets_position": jnp.asarray(pos),
      "targets_segmentation": jnp.asarray(seg),
  }


def _tiny_config(k, **overrides):
  """A tiny dense decoder on one CPU device; eval pbs = N_REAL * k so the eval step sees N_REAL * k rows."""
  kwargs = {
      "run_name": "eval_sample_repeat_test",
      "enable_checkpointing": False,
      "per_device_batch_size": 1.0,
      "eval_per_device_batch_size": float(_N_REAL * k),
      "eval_sample_repeat": k,
      "base_num_decoder_layers": 2,
      "base_emb_dim": 32,
      "base_mlp_dim": 64,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 16,
      "vocab_size": 64,
      "max_target_length": _SEQ,
      "attention": "dot_product",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "scan_layers": False,
      "enable_dropout": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class EvalSampleRepeatConfigTest(unittest.TestCase):
  """types.py derives the real eval count from the repeated batch and validates k."""

  def test_real_count_and_repeated_batch(self):
    for k in (1, 2, 4):
      cfg = _tiny_config(k)
      n_dev = cfg.num_target_devices
      self.assertEqual(cfg.micro_batch_size_to_eval_on, n_dev * _N_REAL * k)
      self.assertEqual(cfg.global_batch_size_to_load_eval, n_dev * _N_REAL * k)
      # global_batch_size_to_eval_on is what the loader treats as real rows and what mllog logs as eval_samples
      # (global_batch_size_to_eval_on * eval_steps): it must not grow with k.
      self.assertEqual(cfg.global_batch_size_to_eval_on, n_dev * _N_REAL)

  def test_rejects_k_below_one(self):
    with self.assertRaises(ValueError):
      _tiny_config(1, eval_sample_repeat=0)

  def test_rejects_k_not_dividing_eval_batch(self):
    # eval batch = 2 * num_devices rows; k=3 does not divide it on 1 or 2^n devices.
    with self.assertRaises(ValueError):
      _tiny_config(1, eval_per_device_batch_size=2.0, eval_sample_repeat=3)


class _StubDecoder(nnx.Module):
  """Per-token logits from an embedding (no cross-row interaction), with the NNX decoder call contract."""

  def __init__(self, vocab_size, hidden, rngs):
    self.embed = nnx.Embed(vocab_size, hidden, rngs=rngs)
    self.proj = nnx.Linear(hidden, vocab_size, rngs=rngs)
    self.mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))

  def __call__(self, decoder_input_tokens, decoder_positions, decoder_segment_ids=None, **kwargs):
    del decoder_positions, decoder_segment_ids, kwargs
    return self.proj(self.embed(decoder_input_tokens))


class _StubCfg:
  """Config subset read by loss_fn / eval_step on the stub model."""

  def __init__(self, k):
    self.model_name = ""
    self.micro_batch_size_to_train_on = _N_REAL
    self.micro_batch_size_to_eval_on = _N_REAL * k
    self.global_batch_size_to_eval_on = _N_REAL
    self.eval_sample_repeat = k
    self.input_data_sharding_logical_axes = ()
    self.vocab_size = 64
    self.z_loss_multiplier = 1e-4
    self.enable_dropout = False
    self.use_multimodal = False
    self.use_indexer = False
    self.indexer_sparse_training = False
    self.indexer_loss_scaling_factor = 0.0
    self.num_vocab_tiling = 1
    self.num_experts = 1
    self.retry_when_tokens_dropped = False
    self.mtp_num_layers = 0
    self.mtp_eval_target_module = 0
    self.use_tunix_gradient_accumulation = False
    self.gradient_accumulation_steps = 1
    self.shard_mode = 0
    self.debug_sharding = False
    self.routed_bias = False
    self.routed_bias_update_rate = 0.0
    self.use_qk_clip = False
    self.weight_sparsity_n = 0
    self.weight_sparsity_m = 0
    self.record_internal_nn_metrics = False


def _scalars(metrics):
  return {key: np.asarray(value) for key, value in metrics["scalar"].items()}


class EvalSampleRepeatStubTest(unittest.TestCase):
  """eval_step on a stub model: k in {1, 2, 4} gives the k=1 metrics, and junk rows never count."""

  def _eval(self, k, data):
    cfg = _StubCfg(k)
    model = _StubDecoder(cfg.vocab_size, hidden=8, rngs=nnx.Rngs(0))
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param))
    graphdef, pure = nnx.split(state)
    return _scalars(pre_train.eval_step(graphdef, cfg, pure, dict(data)))

  def test_metrics_equal_k1(self):
    data = _loaded_batch(64)
    ref = self._eval(1, data)
    expected_weights = int(np.sum(np.asarray(data["targets_segmentation"][:_N_REAL]) != 0))
    self.assertEqual(int(ref["evaluation/total_weights"]), expected_weights)
    for k in (2, 4):
      got = self._eval(k, data)
      self.assertEqual(int(got["evaluation/total_weights"]), int(ref["evaluation/total_weights"]), msg=f"k={k}")
      self.assertEqual(got["evaluation/total_weights"].dtype, ref["evaluation/total_weights"].dtype)
      for key in ("evaluation/loss", "evaluation/total_loss", "evaluation/z_loss"):
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-6, atol=0, err_msg=f"{key} k={k}")

  def test_junk_rows_would_change_the_loss(self):
    """Control: evaluating all loaded rows (the bug a wrong slice would cause) gives a different loss."""
    data = _loaded_batch(64)
    ref = self._eval(1, data)
    cfg = _StubCfg(1)
    cfg.micro_batch_size_to_eval_on = _N_LOADED
    model = _StubDecoder(cfg.vocab_size, hidden=8, rngs=nnx.Rngs(0))
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param))
    graphdef, pure = nnx.split(state)
    wrong = _scalars(pre_train.eval_step(graphdef, cfg, pure, dict(data)))
    self.assertNotEqual(int(wrong["evaluation/total_weights"]), int(ref["evaluation/total_weights"]))
    self.assertGreater(abs(float(wrong["evaluation/loss"]) - float(ref["evaluation/loss"])), 1e-4)

  def test_repeat_eval_samples_tiles_real_rows(self):
    data = _loaded_batch(64)
    cfg = _StubCfg(4)
    out = pre_train.repeat_eval_samples(cfg, data, 4)
    for key, value in out.items():
      self.assertEqual(value.shape[0], _N_REAL * 4)
      for r in range(_N_REAL * 4):
        np.testing.assert_array_equal(np.asarray(value[r]), np.asarray(data[key][r % _N_REAL]), err_msg=key)


class EvalSampleRepeatTinyModelTest(unittest.TestCase):
  """eval_step on a real tiny MaxText decoder (CPU, float32): k in {1, 2, 4} gives the k=1 metrics."""

  def _eval(self, k, data):
    """Runs the jitted eval_step at eval_sample_repeat=k on the loaded batch; returns (scalar metrics, config)."""
    cfg = _tiny_config(k)
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1, aqt=2))
      state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param))
      graphdef, pure = nnx.split(state)
      step = jax.jit(functools.partial(pre_train.eval_step, graphdef, cfg))
      rows = cfg.global_batch_size_to_load_eval
      batch = {key: jnp.concatenate([value] * (rows // _N_LOADED + 1))[:rows] for key, value in data.items()}
      # Only the first global_batch_size_to_eval_on rows are real; every other loaded row is junk.
      return _scalars(step(pure, batch)), cfg

  def test_metrics_equal_k1(self):
    """total_weights exactly equal and loss / total_loss equal to f32 reduction order for k in {2, 4}."""
    if len(jax.devices()) != 1:
      self.skipTest("the tiny-model check builds the batch for one device")
    data = _loaded_batch(64)
    ref, cfg1 = self._eval(1, data)
    n_real = cfg1.global_batch_size_to_eval_on
    expected_weights = int(np.sum(np.asarray(data["targets_segmentation"][:n_real]) != 0))
    self.assertEqual(int(ref["evaluation/total_weights"]), expected_weights)
    self.assertTrue(np.isfinite(ref["evaluation/loss"]))
    for k in (2, 4):
      got, cfg = self._eval(k, data)
      self.assertEqual(cfg.global_batch_size_to_eval_on, n_real)
      self.assertEqual(int(got["evaluation/total_weights"]), int(ref["evaluation/total_weights"]), msg=f"k={k}")
      for key in ("evaluation/loss", "evaluation/total_loss"):
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-5, atol=0, err_msg=f"{key} k={k}")


if __name__ == "__main__":
  unittest.main()
