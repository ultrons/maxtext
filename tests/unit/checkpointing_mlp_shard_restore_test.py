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

"""Params-only restore of an embed-sharded checkpoint under shard_mlp_moe_on_fsdp.

A checkpoint written with the embed sharding of the routed expert weights is chunked along embed,
with the whole mlp dim in each chunk. The flag shards those weights on mlp, so reading them straight
into the flag's sharding makes every device read and decode every chunk (the 512-chip restore
stalled on this). The load must request them in the checkpoint's chunk-aligned sharding and
reshard on device, and return bit-identical values in the flag's sharding.

Runs on 4 devices (CPU: XLA_FLAGS=--xla_force_host_platform_device_count=4).
"""

import contextlib
import os
import tempfile
import unittest
from unittest import mock

from flax import nnx
import jax
import numpy as np
from jax.sharding import Mesh
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

_EXPERT_WEIGHTS = ("wi_0", "wi_1", "wo")


def _cfg(shard_mlp_moe_on_fsdp):
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="mlp_shard_restore_test",
      enable_checkpointing=False,
      model_name="mixtral-8x7b",
      override_model_config=True,
      base_emb_dim=256,
      base_mlp_dim=512,
      base_moe_mlp_dim=512,
      base_num_decoder_layers=2,
      custom_mesh_and_rule="ep-as-dp",
      weight_dtype="float32",
      dtype="bfloat16",
      per_device_batch_size=1,
      max_target_length=64,
      ici_expert_parallelism=1,
      ici_fsdp_parallelism=4,
      sparse_matmul=True,
      megablox=True,
      use_tokamax_gmm=True,
      use_gmm_v2=True,
      shard_embed_moe_on_fsdp=True,
      shard_mlp_moe_on_fsdp=shard_mlp_moe_on_fsdp,
      quantization="fp8_full",
      use_qwix_quantization=True,
      weight_quantization_calibration_method="fixed,-224,224",
      act_quantization_calibration_method="fixed,-224,224",
      bwd_quantization_calibration_method="absmax",
  )


def _kernel_context():
  """Model construction traces a forward through gmm_v2, which needs the Pallas TPU interpreter shim off TPU."""
  if jax.devices()[0].platform == "tpu":
    return contextlib.nullcontext()
  # pylint: disable=import-outside-toplevel
  from tests.unit import pallas_mosaic_tpu_v2_kernel_test as kt

  return kt.cpu_interpret_megablox_v2()


def _abstract_state(config):
  with _kernel_context():
    return _abstract_state_impl(config)


def _abstract_state_impl(config):
  """Abstract TrainStateNNX of `config` on its mesh, as setup_initial_state builds it."""
  mesh = Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
  create_model, model = model_creation_utils.create_nnx_abstract_model(config, mesh)
  tx = optimizers.get_optimizer(config, maxtext_utils.create_learning_rate_schedule(config), model)

  def init_state_fn():
    nnx_model = create_model()
    return train_state_nnx.TrainStateNNX(nnx_model, nnx.Optimizer(nnx_model, tx, wrt=nnx.Param))

  abstract, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, True)
  return abstract


def _flat(tree):
  return {tuple(getattr(k, "key", k) for k in p): v for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _expert_paths(flat):
  return sorted(p for p in flat if p[-1] in _EXPERT_WEIGHTS and len(flat[p].shape) >= 3)


def _params_pure(abstract):
  return checkpointing._abstract_params(abstract).to_pure_dict()  # pylint: disable=protected-access


class MlpShardRestoreTest(unittest.TestCase):
  """save_params_to_path with one sharding -> load_state_if_possible under shard_mlp_moe_on_fsdp."""

  NUM_DEVICES = 4

  def setUp(self):
    super().setUp()
    if jax.device_count() != self.NUM_DEVICES:
      self.skipTest(f"needs exactly {self.NUM_DEVICES} devices")

  def _save(self, config):
    """Writes distinct random values for every weight of `config`'s model in its sharding."""
    abstract = _params_pure(_abstract_state(config))
    leaves, treedef = jax.tree_util.tree_flatten(abstract)
    rng = np.random.default_rng(0)
    values = [jax.device_put(rng.standard_normal(l.shape).astype(l.dtype), l.sharding) if l.shape else l for l in leaves]
    params = jax.tree_util.tree_unflatten(treedef, values)
    root = os.path.join(tempfile.mkdtemp(), "ckpt")
    checkpointing.save_params_to_path(root, {"params": params})  # on disk: params/params/<weights>
    return os.path.join(root, "items"), _flat(params)

  def _restore(self, config, items_path):
    """load_state_if_possible as setup_initial_state calls it; returns (flat restored, flat request)."""
    real_load = checkpointing.ocp.load
    with mock.patch.object(checkpointing.ocp, "load", wraps=real_load) as load:
      _, restored = checkpointing.load_state_if_possible(
          None,
          None,
          items_path,
          "",
          8,
          _abstract_state(config),
          maxtext_config=config,
      )
    self.assertEqual(load.call_count, 1)
    request = _flat(load.call_args.args[1])
    # The request is {restore_key: {collection: weights}}; key it like the restored weights.
    request = {p[2:]: v for p, v in request.items() if p[:2] == ("params", "params")}
    return _flat(restored.to_pure_dict()), request

  def _check_values(self, saved, restored, want):
    self.assertEqual(sorted(saved), sorted(restored))
    for p, v in restored.items():
      np.testing.assert_array_equal(np.asarray(v), np.asarray(saved[p]), err_msg=str(p))
      self.assertEqual(v.sharding, want[p].sharding, msg=str(p))

  def test_embed_sharded_checkpoint_restores_chunk_aligned_under_flag(self):
    items, saved = self._save(_cfg(False))
    off = _flat(_params_pure(_abstract_state(_cfg(False))))
    on = _flat(_params_pure(_abstract_state(_cfg(True))))
    experts = _expert_paths(on)
    self.assertEqual(len(experts), 3)
    for p in experts:
      # The restore sharding of the flag is the embed sharding with the last two entries swapped.
      self.assertNotEqual(on[p].sharding.spec, off[p].sharding.spec, msg=str(p))

    restored, request = self._restore(_cfg(True), items)

    for p, v in request.items():
      want = off[p] if p in experts else on[p]
      self.assertEqual(v.sharding, want.sharding, msg=f"requested sharding of {p}")
    self._check_values(saved, restored, on)

  def test_flag_off_request_is_the_target(self):
    items, saved = self._save(_cfg(False))
    off = _flat(_params_pure(_abstract_state(_cfg(False))))
    restored, request = self._restore(_cfg(False), items)
    for p, v in request.items():
      self.assertEqual(v.sharding, off[p].sharding, msg=str(p))
    self._check_values(saved, restored, off)

  def test_mlp_sharded_checkpoint_restores_directly_under_flag(self):
    # A checkpoint written by a flag-on run is chunked along mlp: the target already reads whole chunks.
    items, saved = self._save(_cfg(True))
    on = _flat(_params_pure(_abstract_state(_cfg(True))))
    restored, request = self._restore(_cfg(True), items)
    for p, v in request.items():
      self.assertEqual(v.sharding, on[p].sharding, msg=str(p))
    self._check_values(saved, restored, on)


if __name__ == "__main__":
  unittest.main()
