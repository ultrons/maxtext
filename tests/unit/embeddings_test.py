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

"""Tests for embeddings.py."""

import sys
import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import embeddings
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


class EmbedTest(unittest.TestCase):
  """Tests for Embed."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    num_embeddings = 100
    num_features = 16
    batch_size = 2
    seq_len = 3

    layer = embeddings.Embed(
        num_embeddings=num_embeddings,
        num_features=num_features,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    inputs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    outputs = layer(inputs)

    self.assertEqual(outputs.shape, (batch_size, seq_len, num_features))

  def test_attend(self):
    num_embeddings = 100
    num_features = 16
    batch_size = 2
    seq_len = 3

    layer = embeddings.Embed(
        num_embeddings=num_embeddings,
        num_features=num_features,
        config=self.cfg,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    query = jnp.ones((batch_size, seq_len, num_features))
    outputs = layer.attend(query)

    self.assertEqual(outputs.shape, (batch_size, seq_len, num_embeddings))


class RotaryEmbeddingTest(unittest.TestCase):
  """Tests for RotaryEmbedding."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.RotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        rngs=self.rngs,
    )

    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])

    outputs = layer(inputs, position=position)

    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 0.988281, 1.38281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class LLaMARotaryEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.LLaMARotaryEmbedding(
        min_timescale=1,
        max_timescale=10000,
        mesh=self.mesh,
        embedding_dims=4,
        use_scale=True,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 1.38281, 0.988281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)


class _YarnPairwiseTestBase(unittest.TestCase):
  """Shared setup for the pairwise YaRN tests: a mesh and a layer factory with the DeepSeek-V3 frequency settings."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)
    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def _layer(self, embedding_dims, **kwargs):
    """A pairwise YaRN layer with the DeepSeek-V3 frequency settings unless overridden."""
    defaults = {
        "max_position_embeddings": 163840,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "rope_theta": 10000.0,
        "rope_factor": 40.0,
        "interleave": True,
        "pairwise": True,
        "rngs": self.rngs,
    }
    defaults.update(kwargs)
    return embeddings.YarnRotaryEmbedding(embedding_dims=embedding_dims, mesh=self.mesh, **defaults)

  @staticmethod
  def _bits(x):
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


class YarnRopeFreqsResidualTest(_YarnPairwiseTestBase):
  """The gathered cos/sin rows can be kept as a `rope_freqs` residual instead of being rematerialized."""

  def test_rope_freqs_saved_under_custom_remat(self):
    """With `rope_freqs` saved, the rematerialized backward does not rebuild the frequency table."""
    layer = self._layer(64)
    inputs = jax.random.normal(jax.random.PRNGKey(6), (1, 64, 2, 64), jnp.float32)
    position = jnp.arange(64, dtype=jnp.int32)[None, :]

    def loss(x):
      return jnp.sum(layer(x, position).astype(jnp.float32) ** 2)

    saved = jax.checkpoint(loss, policy=jax.checkpoint_policies.save_only_these_names("rope_freqs"))
    rematted = jax.checkpoint(loss, policy=jax.checkpoint_policies.nothing_saveable)
    grad_saved = jax.grad(saved)(inputs)
    grad_rematted = jax.grad(rematted)(inputs)
    np.testing.assert_array_equal(self._bits(grad_saved), self._bits(grad_rematted))

    def count_table_ops(fn):
      # Every op that builds the [max_position_embeddings, half_dim] table carries that shape in the jaxpr.
      return str(jax.make_jaxpr(fn)(inputs)).count(f"[{layer.max_position_embeddings},32]")

    n_rematted = count_table_ops(jax.grad(rematted))
    n_saved = count_table_ops(jax.grad(saved))
    self.assertGreater(n_saved, 0)
    self.assertLess(n_saved, n_rematted)

  def test_config_rope_freqs_default_on_device(self):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        remat_policy="custom",
    )
    self.assertIn("rope_freqs", cfg.tensors_on_device)
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        remat_policy="custom",
        rope_freqs="remat",
    )
    self.assertNotIn("rope_freqs", cfg.tensors_on_device)


class YarnRotaryEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(params=0)

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 128,
    }
    argv = [sys.argv[0], get_test_config_path()]
    self.cfg = pyconfig.initialize(argv, **config_arguments)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = jax.sharding.Mesh(devices_array, self.cfg.mesh_axes)

  def test_basic_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Snapshot verification
    expected = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[-0.300781, 0.996094, 1.38281, 1.00781]]]])
    np.testing.assert_allclose(outputs, expected, atol=1e-5)

  def test_pairwise_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=True,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default implementation (pairwise=False, interleave=True)
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=False,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    # Default YaRN RoPE returns concatenated layout [real0, real1, imag0, imag1];
    # pairwise=True returns interleaved layout [real0, imag0, real1, imag1].
    # Convert default concatenated layout to interleaved layout for comparison.
    expected_interleaved = jnp.stack([default_outputs[..., :2], default_outputs[..., 2:]], axis=-1).reshape(
        default_outputs.shape
    )
    np.testing.assert_allclose(outputs, expected_interleaved, atol=1e-5)

  def test_pairwise_requires_interleave(self):
    with self.assertRaises(ValueError):
      embeddings.YarnRotaryEmbedding(
          embedding_dims=4,
          mesh=self.mesh,
          max_position_embeddings=16384,
          original_max_position_embeddings=4096,
          interleave=False,
          pairwise=True,
          rngs=self.rngs,
      )

  def test_non_interleaved_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=False,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default implementation (interleave=True)
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    # For all-ones input, the output of non-interleaved RoPE matches interleaved RoPE
    np.testing.assert_allclose(outputs, default_outputs, atol=1e-5)

  def test_explicit_shard_mode_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        shard_mode=maxtext_utils.ShardMode.EXPLICIT,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default shard_mode (AUTO) implementation
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    np.testing.assert_allclose(outputs, default_outputs, atol=1e-5)

  def test_pairwise_explicit_shard_mode_call(self):
    layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=True,
        shard_mode=maxtext_utils.ShardMode.EXPLICIT,
        rngs=self.rngs,
    )
    inputs = jnp.ones((1, 2, 1, 4))
    position = jnp.array([[0, 1]])
    outputs = layer(inputs, position=position)
    self.assertEqual(outputs.shape, (1, 2, 1, 4))

    # Compare against default implementation (pairwise=False, interleave=True)
    default_layer = embeddings.YarnRotaryEmbedding(
        embedding_dims=4,
        mesh=self.mesh,
        max_position_embeddings=16384,
        original_max_position_embeddings=4096,
        interleave=True,
        pairwise=False,
        shard_mode=maxtext_utils.ShardMode.EXPLICIT,
        rngs=self.rngs,
    )
    default_outputs = default_layer(inputs, position=position)
    # Convert default concatenated layout to interleaved layout for comparison
    expected_interleaved = jnp.stack([default_outputs[..., :2], default_outputs[..., 2:]], axis=-1).reshape(
        default_outputs.shape
    )
    np.testing.assert_allclose(outputs, expected_interleaved, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
