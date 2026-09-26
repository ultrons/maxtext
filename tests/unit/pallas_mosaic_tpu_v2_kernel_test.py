# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Unit tests for Pallas Mosaic TPU v2 kernels."""

import collections
import contextlib
from unittest import mock
import pytest

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental import topologies
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.megablox import common
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_backend
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_backend


def poison_tpu_memory():
  """Fills TPU scratchpad memory with NaNs to simulate garbage state."""
  tpu_info = pltpu.get_tpu_info()
  # Security: Use a large but safe portion of VMEM/SMEM to avoid OOM.
  vmem_size = (4 * 1024 * 1024) // 4  # 4MB
  smem_size = (tpu_info.smem_capacity_bytes // 4) - 8192

  def poison_kernel(in_ref, out_ref, v_scratch, s_scratch):
    del in_ref, out_ref
    v_scratch[...] = jnp.full_like(v_scratch, jnp.nan)
    for i in range(s_scratch.shape[0]):
      s_scratch[i] = 0x7FC00000  # IEEE 754 NaN bit pattern

  pl.pallas_call(
      poison_kernel,
      out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
      grid=(1,),
      scratch_shapes=[
          pltpu.VMEM((vmem_size // 128, 128), jnp.float32),
          pltpu.SMEM((smem_size,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
  )(jnp.zeros((1,), dtype=jnp.float32))


_GroupConfig = collections.namedtuple("_GroupConfig", ["num_groups", "group_offset", "num_local_groups"])


def get_group_sizes(batch_size: int, num_groups: int) -> jax.Array:
  distribution = jax.random.uniform(jax.random.key(0), (num_groups - 1,), dtype=jnp.float32)
  distribution = distribution / jnp.sum(distribution)
  group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
  return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def quantize_tensor(x: jax.Array, dtype: jnp.dtype, axis: int = -1, block_size: int = 256):
  """Quantizes a tensor along a specified axis in blocks."""
  if jnp.issubdtype(dtype, jnp.integer):
    dtype_info = jnp.iinfo(dtype)
    max_val = int(dtype_info.max)
    min_val = int(dtype_info.min)
  else:
    dtype_info = jnp.finfo(dtype)
    max_val = float(dtype_info.max)
    min_val = float(dtype_info.min)

  orig_shape = x.shape
  blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
  x_blocked = x.reshape(blocked_shape)

  x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
  scale = x_blocked_abs_max / max_val
  x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)

  x_q = x_blocked_q.reshape(orig_shape)
  x_q = jnp.nan_to_num(x_q)
  scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
  return x_q, scale


def reference_gmm(
    lhs: jax.Array,  # [m, k]
    rhs: jax.Array,  # [num_groups, k, n]
    group_sizes: jax.Array,  # [num_groups]
    partial_sum: jax.Array | None = None,  # [m, n]
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,  # int32[1]
):
  """Computes reference grouped matrix multiplication."""
  num_tokens = lhs.shape[0]
  num_groups, in_size, out_size = rhs.shape
  assert num_groups > 0, f"rhs must have at least 1 group, got {num_groups}"
  assert lhs.shape[1] == in_size

  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  if rhs_scale is not None:
    num_blocks = rhs_scale.shape[1]
  else:
    num_blocks = 1
  block_size = in_size // num_blocks

  start = 0
  gmm_out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]

    group = global_group - group_offset[0]
    end = min(start + group_size, num_tokens)
    group_size = end - start
    if 0 <= group < num_groups:
      lhs_slice = lhs[start:end]
      rhs_slice = rhs[group]

      out = jnp.array(0.0, dtype=jnp.float32)
      for block in range(num_blocks):
        block_start = block * block_size
        block_end = block_start + block_size
        lhs_block = lhs_slice[:, block_start:block_end].astype(jnp.float32)
        rhs_block = rhs_slice[block_start:block_end, :].astype(jnp.float32)

        acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
        if rhs_scale is not None:
          acc *= rhs_scale[group][block]
        out += acc
      if rhs_bias is not None:
        out = out + rhs_bias[group]
      if partial_sum is not None:
        out = out + partial_sum[start:end]
    else:
      out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)

    gmm_out.append(out.astype(lhs.dtype))
    start = end

  return jnp.concat(gmm_out, axis=0)


def reference_tgmm(
    lhs,  # [k, m]
    rhs,  # [m, n]
    group_sizes,  # [num_groups]
    # num_actual_groups comes from weights.shape[0]
    num_actual_groups,  # int32
    # group_offset is obtained from
    # jnp.arange(0, num_experts, num_experts_per_shard)
    group_offset=None,
    partial_sum=None,
):  # [num_groups, k, n]
  """Computes reference transposed grouped matrix multiplication."""
  # Compute lhs[:, sizes[i-1]:sizes[i]] @ rhs[sizes[i-1]:sizes[i], :]
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  elif jnp.isscalar(group_offset):
    assert group_offset.size == 1
    if jnp.isscalar(group_offset):
      group_offset = group_offset[None]

  start = 0
  out = []
  for global_group in range(group_sizes.size):
    group_size = group_sizes[global_group]
    group = global_group - group_offset[0]
    end = start + group_size
    if 0 <= group < num_actual_groups:
      res = lhs[:, start:end].astype(jnp.float32) @ rhs[start:end, :].astype(jnp.float32)
      if partial_sum is not None:
        res = res + partial_sum[group].astype(jnp.float32)
      out.append(res.astype(lhs.dtype))
    start = end
  return jnp.stack(out)


# Default per-dtype tolerances, mirroring
# jax._src.public_test_util._default_tolerance. Extend this map if a new output
# dtype is introduced into a default-tolerance assertion.
_DTYPE_TOL = {
    jnp.dtype(jnp.bfloat16): 1e-1,
}


def _lookup_tol(dtype):
  key = jnp.dtype(dtype)
  if key not in _DTYPE_TOL:
    raise KeyError(f"No default tolerance for dtype {key!r}. " f"Add it to _DTYPE_TOL or pass explicit atol/rtol.")
  return _DTYPE_TOL[key]


def assert_arrays_all_close(actual, desired, *, atol=None, rtol=None):
  if atol is None:
    atol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  if rtol is None:
    rtol = max(_lookup_tol(actual.dtype), _lookup_tol(desired.dtype))
  chex.assert_trees_all_close(actual, desired, atol=atol, rtol=rtol)


class GmmTest(parameterized.TestCase):

  def setUp(self):
    if jax.default_backend() != "tpu":
      self.skipTest("Only supported on TPUs.")
    super().setUp()

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 512],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      has_bias=[True, False],
      has_partial_sum=[True, False],
      group_offset=[0, 2, 3],
  )
  def test_gmm_basic(self, batch_size, in_size, out_size, num_groups, has_bias, has_partial_sum, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2, k3 = jax.random.split(key, 4)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(k2, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)
    ps = None
    if has_partial_sum:
      ps = jax.random.normal(k3, (batch_size, out_size), dtype=jnp.bfloat16)

    expected = reference_gmm(lhs, rhs, group_sizes, partial_sum=ps, rhs_bias=rhs_bias, group_offset=group_offset)

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        partial_sum=ps,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
  )
  def test_gmm_partial_sum(self, batch_size, in_size, out_size, num_groups, group_offset):
    """Test GMM with partial sum accumulation and memory aliasing."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    k0, k1, k2 = jax.random.split(key, 3)

    lhs = jax.random.normal(k0, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k1, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)
    ps = jax.random.normal(k2, (batch_size, out_size), dtype=jnp.bfloat16)

    expected = reference_gmm(lhs, rhs, group_sizes, partial_sum=ps, group_offset=group_offset)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        partial_sum=ps,
        group_offset=group_offset,
    )
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 1024],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[5, 16, 32],
      has_partial_sum=[True, False],
      group_offset=[0, 2, 3],
  )
  def test_tgmm_basic(self, batch_size, in_size, out_size, num_groups, has_partial_sum, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)  # [m, k]
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)  # [m, n]
    group_sizes = get_group_sizes(batch_size, num_groups)
    # if batch_size=128, num_groups=3, an example group_size is
    # group_sizes=Array([14, 14, ..., 7]).
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    ps = None
    if has_partial_sum:
      ps = jax.random.normal(key3, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)

    lhs_t = lhs.swapaxes(0, 1)  # [k, m]
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset, partial_sum=ps)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        partial_sum=ps,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    # diff = jnp.abs(expected - actual)
    # max_diff_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
    # print(f"Output max diff: {jnp.max(diff)} at index {max_diff_idx}")
    # print(f"Output mean diff: {jnp.mean(jnp.abs(expected - actual))}")
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255, 500],
      out_size=[255, 500],
      num_groups=[16],
      group_offset=[0],
  )
  def test_tgmm_implicit_padding(self, batch_size, in_size, out_size, num_groups, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[256, 1024],
      in_size=[1024],
      out_size=[1024],
      num_groups=[16],
      group_offset=[0, 2],
      tile_k=[256, 512],
      tile_n=[256, 512],
  )
  def test_tgmm_with_tile_info(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      tile_k,
      tile_n,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)

    tile_info = gmm_backend.TileSizes(tile_m=256, tile_k=tile_k, tile_n=tile_n)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1, 2, 3],
  )
  def test_tgmm_empty_group(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly zeros output for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    # Redistribute the empty group's tokens to the last group.
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)

    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[256],
      in_size=[512],
      out_size=[512],
      num_groups=[4],
      group_offset=[0],
      empty_group_index=[0, 1, 2, 3],
  )
  def test_tgmm_empty_group_with_partial_sum(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      group_offset,
      empty_group_index,
  ):
    """Test that TGMM correctly preserves partial sum for empty groups."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)
    key1, key2, key3 = jax.random.split(key, 3)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    ps = jax.random.normal(key3, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_sizes = group_sizes.at[-1].add(group_sizes[empty_group_index])
    group_sizes = group_sizes.at[empty_group_index].set(0)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset, partial_sum=ps)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        partial_sum=ps,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  def test_tgmm_explicitly_exercises_all_branches(self):
    # Group 0 (size 4*tile_m, 4 gm tiles): matmul_new_group, matmul, matmul,
    # matmul_group_changing.
    # Group 1 (size 64, 1 gm tile): matmul_new_group_and_changing.

    tile_m = tile_k = tile_n = 256
    in_size = out_size = 256
    num_local_groups = 2
    g0, g1 = 4 * tile_m, 64
    batch_size = g0 + g1

    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)
    lhs = jax.random.normal(key1, (batch_size, in_size), dtype=jnp.bfloat16)
    grad = jax.random.normal(key2, (batch_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([g0, g1], dtype=jnp.int32)
    group_offset = jnp.array(0, dtype=jnp.int32)

    lhs_t = lhs.swapaxes(0, 1)
    expected = reference_tgmm(lhs_t, grad, group_sizes, num_local_groups, group_offset=group_offset)
    tile_info = gmm_backend.TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)
    actual = tgmm_backend.tgmm_v2(
        lhs,
        grad,
        group_sizes,
        num_local_groups,
        group_offset=group_offset,
        preferred_element_type=jnp.bfloat16,
        tile_info=tile_info,
    )
    self.assertEqual(actual.shape, (num_local_groups, in_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      has_bias=[True, False],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[64, 128, 256, 512],
      group_offset=[0, 2, 3],
  )
  def test_gmm_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and common.tpu_generation() < 7:
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  def test_gmm_security_isolation(self):
    """Verifies that sequences (experts) are isolated from each other.

    This test checks that NaNs or extreme values in one expert group do not
    pollute the output of other expert groups, even if they share the same
    sublane tile.
    """
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(42)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_groups, in_size, out_size), dtype=jnp.bfloat16)

    # We use very small group sizes to force expert groups to share tiles.
    # sublane_size is typically 8 or 16.
    group_sizes = jnp.array([4, 4, 4, batch_size - 12], dtype=jnp.int32)

    # 1. Run baseline
    actual_clean = gmm_backend.gmm_v2(lhs, rhs, group_sizes)

    # 2. Inject NaNs into all experts except the first one.
    # If isolation fails, the NaNs will leak into the first expert's output.
    rhs_malicious = rhs.at[1:].set(jnp.nan)
    actual_malicious = gmm_backend.gmm_v2(lhs, rhs_malicious, group_sizes)

    # Verify that the first expert's output is identical and NaN-free.
    first_expert_size = group_sizes[0]
    chex.assert_trees_all_close(
        actual_malicious[:first_expert_size],
        actual_clean[:first_expert_size],
        atol=0.0,
        rtol=0.0,
    )
    self.assertFalse(jnp.any(jnp.isnan(actual_malicious[:first_expert_size])))

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  def test_gmm_uninitialized_memory_robustness(self):
    """Verifies that the kernel is robust against uninitialized scratchpads.

    This test intentionally poisons TPU VMEM/SMEM with NaNs before running the
    GMM kernel. This ensures that  no stale data from previous sessions can leak
    into the output.
    """
    # 1. Poison TPU memory with NaNs
    poison_tpu_memory()

    # 2. Run GMM kernel
    batch_size = 128
    in_size = 512
    out_size = 512
    num_groups = 4
    key = jax.random.key(0)
    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_groups, in_size, out_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([batch_size // 4] * 4, dtype=jnp.int32)

    actual = gmm_backend.gmm_v2(lhs, rhs, group_sizes)

    # 3. Verify that the output is NaN-free
    self.assertFalse(jnp.any(jnp.isnan(actual)))

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn, jnp.float4_e2m1fn],
      block_size=[1024],
      tile_k=[128, 256, 512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test that quant_block_size > tile_k is handled correctly."""
    if weight_dtype == jnp.float4_e2m1fn and common.tpu_generation() < 7:
      self.skipTest("Expect TPUv7+")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_backend.TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[1024],
      out_size=[512],
      num_groups=[16],
      weight_dtype=[jnp.int4, jnp.int8, jnp.float8_e4m3fn],
      block_size=[1024],
      tile_k=[128, 256, 512],
      group_offset=[0],
  )
  def test_gmm_activation_weight_quantized_block_larger_than_tile_k(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      tile_k,
      group_offset,
  ):
    """Test activation+weight quantized path with quant_block_size > tile_k."""
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    tile_info = gmm_backend.TileSizes(tile_m=128, tile_k=tile_k, tile_n=out_size)
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        tile_info=tile_info,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.2, rtol=1.2)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512, 1024],
      out_size=[512, 1024],
      num_groups=[16, 32],
      weight_dtype=[jnp.int4, jnp.uint4, jnp.int8, jnp.float8_e4m3fn],
      block_size=[512, 1024],
      group_offset=[0, 2, 3],
  )
  def test_gmm_activation_weight_quantized(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      weight_dtype,
      block_size,
      group_offset,
  ):
    if weight_dtype == jnp.float4_e2m1fn and common.tpu_generation() < 7:
      self.skipTest("Expect TPUv7+")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        maybe_quantize_lhs=True,
    ).astype(lhs.dtype)

    chex.assert_trees_all_close(actual, expected, atol=1.1, rtol=1.1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128, 256],
      in_size=[255, 500],
      out_size=[255, 500],
      num_groups=[16],
      has_bias=[True, False],
      group_offset=[0],
  )
  def test_gmm_implicit_padding(self, batch_size, in_size, out_size, num_groups, has_bias, group_offset):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[500],
      num_groups=[16],
      has_bias=[True, False],
      weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
      block_size=[512],
      group_offset=[0],
  )
  def test_gmm_weight_quantized_padding(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      weight_dtype,
      block_size,
      group_offset,
  ):
    num_local_groups = num_groups - group_offset
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_q, rhs_scale = quantize_tensor(rhs, weight_dtype, axis=1, block_size=block_size)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        group_offset=group_offset,
        rhs_bias=rhs_bias,
        maybe_quantize_lhs=False,
    ).astype(lhs.dtype)

    self.assertEqual(actual.shape, (batch_size, out_size))
    chex.assert_trees_all_close(actual, expected, atol=3e-1, rtol=3e-1)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      # group_config: (num_groups, group_offset, num_local_groups)
      group_config=[
          # groups 0-1: group<0, groups 2-5: local and active,
          # groups 6-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=2, num_local_groups=4),
          # no negative groups, groups 0-7: local and active,
          # groups 8-15: group>=num_local_groups
          _GroupConfig(num_groups=16, group_offset=0, num_local_groups=8),
          # groups 0-3: group<0, groups 4-7: local and active,
          # groups 8-31: group>=num_local_groups
          _GroupConfig(num_groups=32, group_offset=4, num_local_groups=4),
      ],
  )
  def test_gmm_nonlocal_groups_produce_zeros(self, batch_size, in_size, out_size, group_config):
    num_groups, group_offset, num_local_groups = group_config
    key = jax.random.key(0)

    lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16)
    rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array(group_offset, dtype=jnp.int32)

    expected = reference_gmm(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    actual = gmm_backend.gmm_v2(
        lhs,
        rhs,
        group_sizes,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    self.assertEqual(actual.shape, (batch_size, out_size))
    assert_arrays_all_close(actual, expected)

  @pytest.mark.skip(reason="Test takes too long, can run locally to verify changes b/528087469")
  @parameterized.product(
      batch_size=[128],
      in_size=[512],
      out_size=[512],
      num_groups=[16],
      has_bias=[True, False],
      use_weight_scale=[True, False],
      maybe_quantize_lhs=[True, False],
      fuse_act=["silu", "swigluoai", "gelu"],
      group_offset=[0, 2],
      block_size=[256, 512],
  )
  def test_gmm_fused_activation(
      self,
      batch_size,
      in_size,
      out_size,
      num_groups,
      has_bias,
      use_weight_scale,
      maybe_quantize_lhs,
      fuse_act,
      group_offset,
      block_size,
  ):
    if maybe_quantize_lhs and not use_weight_scale:
      self.skipTest("LHS quantization requires RHS quantization/scale in this config.")
    if block_size > in_size:
      self.skipTest("block_size must be <= in_size")
    key = jax.random.key(0)
    final_out_size = out_size // 2
    num_local_groups = num_groups - group_offset

    # 1. Generate Inputs
    lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
    rhs = jax.random.uniform(key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1)

    rhs_q = rhs
    rhs_scale = None
    if use_weight_scale:
      rhs_q, rhs_scale = quantize_tensor(rhs, jnp.int8, axis=1, block_size=block_size)
      rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

    rhs_bias = None
    if has_bias:
      rhs_bias = jax.random.normal(key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16)

    group_sizes = get_group_sizes(batch_size, num_groups)
    group_offset = jnp.array([group_offset], dtype=jnp.int32)

    # 2. Simulate LHS Quantization Noise
    lhs_simulated = lhs
    # because the kernel quantizes LHS in blocks, while reference does it at the
    # whole tensor level, and output is casted down we need to simulate that
    # quantization noise in the reference as well for a fair comparison
    if maybe_quantize_lhs:
      lhs_block_size = min(512, in_size)
      lhs_q, lhs_scale_factor = quantize_tensor(lhs, jnp.int8, axis=1, block_size=lhs_block_size)
      lhs_q_blocked = lhs_q.reshape(batch_size, -1, lhs_block_size).astype(jnp.float32)
      lhs_scale_expanded = jnp.expand_dims(lhs_scale_factor, axis=2)
      lhs_simulated = (lhs_q_blocked * lhs_scale_expanded).reshape(lhs.shape).astype(lhs.dtype)

    # 3. Compute Reference Output
    raw_expected = reference_gmm(
        lhs_simulated,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
    )

    # Slice the reference and apply the activation function
    expected = gmm_backend.apply_act_fn(raw_expected.astype(jnp.float32), fuse_act).astype(lhs.dtype)

    # 4. Compute Actual Kernel Output
    actual = gmm_backend.gmm_v2(
        lhs,
        rhs_q,
        group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        maybe_quantize_lhs=maybe_quantize_lhs,
        fuse_act=fuse_act,
    ).astype(lhs.dtype)

    # 5. Compare Results
    self.assertEqual(actual.shape, (batch_size, final_out_size))

    # tolerances based quantization noise difference between reference and
    # gmm_v2
    if maybe_quantize_lhs:
      atol, rtol = 4.0, 2.0  # Act + Weight Quantization
    elif use_weight_scale:
      atol, rtol = 3e-1, 3e-1  # Weight Quantization Only
    else:
      atol, rtol = 5e-2, 5e-2  # Unquantized Path (bfloat16 precision diffs)

    chex.assert_trees_all_close(actual, expected, atol=atol, rtol=rtol)


# ==============================================================================
# transpose_rhs: gmm_v2 on a [g, n, k] rhs must match gmm_v2 on rhs.swapaxes(1, 2) bit for bit.
# ==============================================================================


def _interpret_internals():
  """Returns the jax internals the CPU interpreter shim patches (raises if this jax version lacks them)."""
  # pylint: disable=import-outside-toplevel,protected-access
  from jax._src import tpu_info
  from jax._src.pallas.mosaic.interpret import shared_memory
  from jax._src.pallas.mosaic.interpret import utils as interpret_utils
  from jax._src.state import types as state_types

  for attr in ("registry", "_get_tpu_info_impl", "ChipVersion", "get_tpu_info"):
    getattr(tpu_info, attr)
  for attr in ("to_range", "_transform_slice_or_index", "_compose_slice_or_index"):
    getattr(interpret_utils, attr)
  getattr(shared_memory.SharedMemory, "store_buffer_content")
  getattr(state_types, "ReshapeTransform")
  return tpu_info, shared_memory, interpret_utils, state_types


@contextlib.contextmanager
def _cpu_interpret_gmm_v2(internals):
  """Runs gmm_v2 under the Pallas TPU interpreter on a non-TPU backend.

  The kernel DMAs from `ref.reshape(rows // sublanes, sublanes, cols)` views of
  its 2-D HBM refs. The interpreter only folds NDIndexer transforms into a
  numpy range, so this context folds a ReshapeTransform followed by an indexer
  whose sublane axis is fully selected back into a row range on the original
  buffer, and reshapes DMA payloads to the destination range at the store. It
  also registers the TPU7x hardware info for the "cpu" device kind so that the
  kernel's tiling logic can run. Everything is restored on exit.
  """
  # pylint: disable=protected-access
  tpu_info, shared_memory, interpret_utils, state_types = internals

  def to_range(transforms):
    ret = ()
    pending = None
    for transform in transforms:
      if isinstance(transform, state_types.ReshapeTransform):
        assert pending is None
        pending = tuple(int(d) for d in transform.shape)
        continue
      idx = tuple(interpret_utils._transform_slice_or_index(i) for i in transform.indices)
      if pending is not None:
        sublanes = pending[1]
        idx = idx + tuple(slice(0, d, 1) for d in pending[len(idx) :])
        assert idx[1] == slice(0, sublanes, 1), (idx, pending)
        i0 = idx[0]
        if isinstance(i0, int):
          rows = slice(i0 * sublanes, (i0 + 1) * sublanes, 1)
        else:
          assert i0.step == 1
          rows = slice(i0.start * sublanes, i0.stop * sublanes, 1)
        idx = (rows,) + tuple(idx[2:])
        pending = None
      ret = interpret_utils._compose_slice_or_index(ret, idx)
    return ret

  orig_store = shared_memory.SharedMemory.store_buffer_content

  def store_buffer_content(self, key, rnge, value, *args, **kwargs):
    target = tuple((r.stop - r.start) // (r.step or 1) for r in rnge if isinstance(r, slice))
    value = np.asarray(value)
    if rnge and value.shape != target and value.size == int(np.prod(target)):
      value = value.reshape(target)
    return orig_store(self, key, rnge, value, *args, **kwargs)

  orig_pallas_call = pl.pallas_call

  def pallas_call(*args, **kwargs):
    kwargs["interpret"] = pltpu.InterpretParams(uninitialized_memory="zero")
    return orig_pallas_call(*args, **kwargs)

  orig_to_range = interpret_utils.to_range
  had_cpu_entry = "cpu" in tpu_info.registry
  tpu_info.registry["cpu"] = lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1)
  interpret_utils.to_range = to_range
  shared_memory.SharedMemory.store_buffer_content = store_buffer_content
  pl.pallas_call = pallas_call
  try:
    yield
  finally:
    pl.pallas_call = orig_pallas_call
    shared_memory.SharedMemory.store_buffer_content = orig_store
    interpret_utils.to_range = orig_to_range
    if not had_cpu_entry:
      del tpu_info.registry["cpu"]
    cache_clear = getattr(tpu_info.get_tpu_info, "cache_clear", None)
    if cache_clear is not None:
      cache_clear()


def _reference_gmm_transposed(lhs, rhs_gnk, group_sizes, group_offset, rhs_scale=None, lhs_scale=None, lhs_qtype=None):
  """f32 reference of lhs[m, k] @ rhs_gnk[g, n, k].T per group; groups outside the local range give zeros."""
  m = lhs.shape[0]
  num_local_groups, n, _ = rhs_gnk.shape
  lhs = np.asarray(lhs).astype(np.float32)
  if lhs_qtype is not None:
    # Same per-tensor fixed-scale quantization the kernel applies in VMEM.
    qmax = float(jnp.finfo(lhs_qtype).max)
    scale = float(np.asarray(lhs_scale).reshape(()))
    lhs_q = jnp.clip(jnp.asarray(lhs) / scale, -qmax, qmax).astype(lhs_qtype)
    lhs = np.asarray(lhs_q).astype(np.float32) * scale
  rhs = np.asarray(rhs_gnk).astype(np.float32)
  out = np.zeros((m, n), np.float32)
  start = 0
  for global_group in range(group_sizes.shape[0]):
    end = start + int(group_sizes[global_group])
    local_group = global_group - group_offset
    if 0 <= local_group < num_local_groups and end > start:
      acc = lhs[start:end] @ rhs[local_group].T
      if rhs_scale is not None:
        acc = acc * np.asarray(rhs_scale[local_group, 0, 0]).astype(np.float32)
      out[start:end] = acc
    start = end
  return out


# group sizes over 6 lhs groups; with group_offset=1 the four local groups have sizes 0, 91, 0, 60 (two empty), and the
# group boundaries are not sublane aligned.
_TRHS_GROUP_SIZES = (37, 0, 91, 0, 60, 68)
# 16 local groups with several empty ones, for the production-like 16-expert shape.
_TRHS_GROUP_SIZES_16 = (0, 40, 0, 96, 13, 0, 0, 77, 30, 0, 64, 5, 0, 99, 88, 0)


class GmmTransposeRhsTest(parameterized.TestCase):
  """gmm_v2(transpose_rhs=True) reads the [g, n, k] weight in place; its output must match the [g, k, n] path bit for bit.

  Runs natively on TPU. On other backends it runs both kernels under the Pallas TPU interpreter (see
  `_cpu_interpret_gmm_v2`), which checks the index maps, masking and scale handling but not the MXU numerics.
  """

  def _kernel_context(self):
    if jax.default_backend() == "tpu":
      return contextlib.nullcontext()
    try:
      internals = _interpret_internals()
    except (ImportError, AttributeError) as e:
      self.skipTest(f"Pallas TPU interpreter shim does not fit this jax version: {e!r}")
    return _cpu_interpret_gmm_v2(internals)

  @parameterized.named_parameters(
      # name, m, k, n, num_local_groups, group_sizes, group_offset, lhs dtype, rhs dtype, tiles, quantize_lhs
      ("bf16", 256, 512, 384, 4, _TRHS_GROUP_SIZES, 1, jnp.bfloat16, jnp.bfloat16, (128, 256, 256), False),
      # Production dlhs: e5m2 gradient qvalue against the e4m3 weight qvalue, no in-kernel quantization.
      ("fp8_dlhs", 256, 512, 384, 4, _TRHS_GROUP_SIZES, 1, jnp.float8_e5m2, jnp.float8_e4m3fn, (128, 256, 256), False),
      # bf16 lhs quantized in VMEM with a fixed per-tensor scale against an e4m3 weight with a per-channel scale.
      (
          "fp8_quantize_lhs",
          256,
          512,
          384,
          4,
          _TRHS_GROUP_SIZES,
          1,
          jnp.bfloat16,
          jnp.float8_e4m3fn,
          (128, 512, 256),
          True,
      ),
      # size_k % tile_k != 0 exercises the valid_k mask, which sits on the minor axis of the transposed rhs tile.
      ("bf16_ragged_k", 256, 384, 256, 4, _TRHS_GROUP_SIZES, 1, jnp.bfloat16, jnp.bfloat16, (128, 256, 256), False),
      (
          "fp8_ragged_k",
          256,
          384,
          256,
          4,
          _TRHS_GROUP_SIZES,
          1,
          jnp.float8_e5m2,
          jnp.float8_e4m3fn,
          (128, 256, 256),
          False,
      ),
      # wi dlhs of the 512-chip DeepSeek-V3 recipe at reduced m: k = mlp 2048, n = embed tile 3584, 16 experts.
      (
          "fp8_dlhs_16_experts",
          512,
          2048,
          3584,
          16,
          _TRHS_GROUP_SIZES_16,
          0,
          jnp.float8_e5m2,
          jnp.float8_e4m3fn,
          (256, 2048, 3584),
          False,
      ),
  )
  def test_matches_swapaxes_path(
      self, m, k, n, num_local_groups, group_sizes, group_offset, lhs_dtype, rhs_dtype, tiles, quantize_lhs
  ):
    key = jax.random.key(1)
    key_lhs, key_rhs, key_scale = jax.random.split(key, 3)
    lhs = jax.random.normal(key_lhs, (m, k), jnp.float32).astype(lhs_dtype)
    # The weight as the caller holds it: [g, n, k] (e.g. the forward [g, k_fwd, n_fwd] weight seen from the dlhs
    # matmul, which contracts over n_fwd).
    rhs_gnk = jax.random.normal(key_rhs, (num_local_groups, n, k), jnp.float32).astype(rhs_dtype)
    rhs_scale = lhs_scale = lhs_qtype = None
    if quantize_lhs:
      rhs_scale = jax.random.uniform(key_scale, (num_local_groups, 1, 1, n), jnp.float32, 0.5, 1.5)
      lhs_scale = jnp.full((1, 1), 2.0, jnp.float32)
      lhs_qtype = jnp.float8_e4m3fn
    group_sizes = jnp.asarray(group_sizes, jnp.int32)
    group_offset = jnp.array([group_offset], jnp.int32)
    kwargs = {
        "group_sizes": group_sizes,
        "group_offset": group_offset,
        "rhs_scale": rhs_scale,
        "lhs_scale": lhs_scale,
        "tile_info": gmm_backend.TileSizes(*tiles),
        "maybe_quantize_lhs": quantize_lhs,
        "preferred_element_type": jnp.bfloat16,
    }

    with self._kernel_context():
      expected = gmm_backend.gmm_v2(lhs, rhs_gnk.swapaxes(1, 2), **kwargs)
      actual = gmm_backend.gmm_v2(lhs, rhs_gnk, transpose_rhs=True, **kwargs)
      expected, actual = jax.block_until_ready((expected, actual))

    self.assertEqual(actual.shape, (m, n))
    self.assertEqual(actual.dtype, expected.dtype)
    max_abs_diff = float(jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))))
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        np.asarray(expected).view(np.uint16),
        err_msg=f"transpose_rhs output differs from the swapaxes path, max abs diff {max_abs_diff}",
    )
    # Sanity check against an f32 reference, so that a shared bug cannot hide behind the equality above.
    reference = _reference_gmm_transposed(
        lhs, rhs_gnk, group_sizes, int(group_offset[0]), rhs_scale, lhs_scale, lhs_qtype
    )
    chex.assert_trees_all_close(actual.astype(jnp.float32), jnp.asarray(reference), atol=0.5, rtol=5e-2)

  def test_dlhs_switch_selects_kernel_and_matches(self):
    """ops._dlhs_run_tokamax_v2 gives the same dlhs with the in-kernel transpose on and off (bit for bit off-TPU)."""
    m, k, n, num_groups = 256, 512, 384, 4
    key_lhs, key_rhs = jax.random.split(jax.random.key(2))
    dout = jax.random.normal(key_lhs, (m, n), jnp.float32).astype(jnp.bfloat16)  # [m, n]
    rhs_gkn = jax.random.normal(key_rhs, (num_groups, k, n), jnp.float32).astype(jnp.bfloat16)  # fwd weight [g, k, n]
    group_sizes = jnp.array([100, 0, 60, 96], jnp.int32)
    tiling = (128, 128, 128, 128, 256, 256, 128, 128, 128)  # dlhs tiles are tiling[3:6]
    outs = {}
    with self._kernel_context():
      for use_kernel in (False, True):
        with mock.patch.object(megablox_ops, "DLHS_USE_TRANSPOSED_RHS_KERNEL", use_kernel):
          with mock.patch.object(gmm_backend, "gmm_v2", wraps=gmm_backend.gmm_v2) as spy:
            outs[use_kernel] = jax.block_until_ready(
                megablox_ops._dlhs_run_tokamax_v2(  # pylint: disable=protected-access
                    dout, rhs_gkn, group_sizes, None, jnp.bfloat16, tiling, False, False
                )
            )
            self.assertEqual(spy.call_args.kwargs["transpose_rhs"], use_kernel)
            self.assertEqual(spy.call_args.kwargs["rhs"].shape, (num_groups, k, n) if use_kernel else (num_groups, n, k))
    np.testing.assert_array_equal(np.asarray(outs[True]).view(np.uint16), np.asarray(outs[False]).view(np.uint16))

  def test_rejects_fuse_act(self):
    lhs = jnp.zeros((256, 512), jnp.bfloat16)
    rhs_gnk = jnp.zeros((4, 512, 512), jnp.bfloat16)
    group_sizes = jnp.array([64, 64, 64, 64], jnp.int32)
    with self._kernel_context():
      with self.assertRaises(NotImplementedError):
        gmm_backend.gmm_v2(
            lhs,
            rhs_gnk,
            group_sizes,
            tile_info=gmm_backend.TileSizes(128, 256, 256),
            fuse_act="silu",
            transpose_rhs=True,
        )


class GmmTransposeRhsCompileTest(parameterized.TestCase):
  """Mosaic compiles the transposed-rhs kernel for tpu7x (virtual topology, no hardware)."""

  @pytest.mark.tpu_backend
  @parameterized.named_parameters(
      # Production dlhs kernels of the 512-chip DeepSeek-V3 recipe (fp8_full, fixed calibration): lhs is the e5m2
      # gradient qvalue, rhs the e4m3 weight qvalue, tiles from wi_tile_dlhs_* / wo_tile_dlhs_*.
      ("wi_dlhs_fp8", 40960, 2048, 7168, 16, jnp.float8_e5m2, jnp.float8_e4m3fn, (256, 2048, 3584)),
      ("wo_dlhs_fp8", 40960, 7168, 2048, 16, jnp.float8_e5m2, jnp.float8_e4m3fn, (512, 1792, 2048)),
      ("bf16", 1024, 512, 384, 4, jnp.bfloat16, jnp.bfloat16, (128, 256, 256)),
  )
  def test_compiles_for_tpu7x(self, m, k, n, num_groups, lhs_dtype, rhs_dtype, tiles):
    try:
      topology = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"tpu7x virtual topology unavailable (needs a TPU-enabled jax install): {e!r}")

    lhs = jax.ShapeDtypeStruct((m, k), lhs_dtype)
    rhs_gnk = jax.ShapeDtypeStruct((num_groups, n, k), rhs_dtype)
    group_sizes = jax.ShapeDtypeStruct((num_groups,), jnp.int32)

    def dlhs(lhs, rhs, group_sizes):
      return gmm_backend.gmm_v2(
          lhs,
          rhs,
          group_sizes,
          tile_info=gmm_backend.TileSizes(*tiles),
          maybe_quantize_lhs=False,
          preferred_element_type=jnp.bfloat16,
          transpose_rhs=True,
      )

    with jax.default_device(topology.devices[0]):
      compiled = jax.jit(dlhs).lower(lhs, rhs_gnk, group_sizes).compile()
    self.assertIn("gmm_v2", compiled.as_text())

  @pytest.mark.tpu_backend
  def test_accepts_8_row_tiled_rhs_without_copy(self):
    """The production operand: the FSDP-gathered fp8 weight arrives in {2,1,0:T(8,128)(4,1)} and XLA feeds it as is.

    The Pallas custom call constrains only the dimension order of its operands, so an 8-row-tiled rhs (what the
    SparseCore all-gather produces when the per-shard row count is not a multiple of 32, e.g. 7168 / 64) reaches the
    kernel without a re-tiling copy and Mosaic lowers the tile DMAs against that tiling. This documents the condition
    under which the transposed-rhs dlhs kernel runs in the 512-chip program; see ops.DLHS_USE_TRANSPOSED_RHS_KERNEL.
    """
    try:
      topology = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
      from jax.experimental.layout import Format, Layout  # pylint: disable=import-outside-toplevel
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"tpu7x virtual topology or layout API unavailable: {e!r}")

    m, k, n, num_groups = 40960, 2048, 7168, 16
    lhs = jax.ShapeDtypeStruct((m, k), jnp.float8_e5m2)
    rhs_gnk = jax.ShapeDtypeStruct((num_groups, n, k), jnp.float8_e4m3fn)
    group_sizes = jax.ShapeDtypeStruct((num_groups,), jnp.int32)
    rhs_format = Format(
        Layout(major_to_minor=(0, 1, 2), tiling=((8, 128), (4, 1))),
        jax.sharding.SingleDeviceSharding(topology.devices[0]),
    )

    def dlhs(lhs, rhs, group_sizes):
      return gmm_backend.gmm_v2(
          lhs,
          rhs,
          group_sizes,
          tile_info=gmm_backend.TileSizes(256, 2048, 3584),
          maybe_quantize_lhs=False,
          preferred_element_type=jnp.bfloat16,
          transpose_rhs=True,
      )

    with jax.default_device(topology.devices[0]):
      compiled = jax.jit(dlhs, in_shardings=(None, rhs_format, None)).lower(lhs, rhs_gnk, group_sizes).compile()
    hlo = compiled.as_text()
    self.assertIn("f8e4m3fn[16,7168,2048]{2,1,0:T(8,128)(4,1)} parameter", hlo)
    self.assertNotRegex(hlo, r"= f8e4m3fn\[16,7168,2048\][^ ]* copy\(")


if __name__ == "__main__":
  absltest.main()
