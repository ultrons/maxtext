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

"""Tests for the pairwise RoPE Pallas kernel: bit equality with the reshape form, forward and backward."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from maxtext.kernels import rope_pairwise

INTERPRET = jax.default_backend() != "tpu"


def reference_pairwise(inputs, cos_rows, sin_rows, out_dtype, scale=1.0):
  """The reshape form of `YarnRotaryEmbedding` (pairs, flip, rotate), on per-pair cos/sin rows [B, S, H // 2]."""
  b, s, n, h = inputs.shape
  pairs = inputs.reshape(b, s, n, h // 2, 2).astype(jnp.float32)
  cos = cos_rows.astype(jnp.float32)[:, :, None, :, None]
  sin = sin_rows.astype(jnp.float32)[:, :, None, :, None]
  swapped = jnp.flip(pairs, axis=-1)
  sign = jnp.asarray([-1.0, 1.0], dtype=jnp.float32)
  output = (pairs * cos + swapped * sin * sign).reshape(b, s, n, h)
  if scale != 1.0:
    output = output * scale
  return output.astype(out_dtype)


def bits(x):
  x = np.asarray(x)
  return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def legal_evaluations(x, cos_rows, sin_rows, out_dtype, scale=1.0, negate_sin=False, scale_first=False):
  """The two IEEE-legal f32 evaluations of `x * cos + swap(x) * (sin * sign)` (then `* scale`, cast), in numpy.

  `strict` rounds every operation to f32; the two `fma` variants fuse one of the multiplies into the add with a
  single rounding, which is what XLA:CPU's LLVM backend does to any compiled multiply-add (its target options
  always allow FMA fusion). An execution of the same expression on any backend must equal one of the three at
  every element. With `negate_sin` and
  `scale_first` the same is computed for the backward kernel (cotangent in, `sin * sign` negated, scale applied first).
  """
  h = x.shape[-1]
  xn = np.asarray(jnp.asarray(x).astype(jnp.float32)).astype(np.float32)
  if scale_first and scale != 1.0:
    xn = (xn * np.float32(scale)).astype(np.float32)
  cos = np.repeat(np.asarray(cos_rows, np.float32), 2, axis=-1)[:, :, None, :]
  sin = np.repeat(np.asarray(sin_rows, np.float32), 2, axis=-1)[:, :, None, :]
  sign = np.tile(np.array([-1.0, 1.0], np.float32), h // 2)
  ss = (sin * sign).astype(np.float32)
  if negate_sin:
    ss = -ss
  swapped = xn.copy()
  swapped[..., 0::2] = xn[..., 1::2]
  swapped[..., 1::2] = xn[..., 0::2]
  a = (xn * cos).astype(np.float32)
  t = (swapped * ss).astype(np.float32)
  strict = (a + t).astype(np.float32)
  fma_x = (xn.astype(np.float64) * cos.astype(np.float64) + t.astype(np.float64)).astype(np.float32)
  fma_sw = (swapped.astype(np.float64) * ss.astype(np.float64) + a.astype(np.float64)).astype(np.float32)
  outs = [strict, fma_x, fma_sw]
  if not scale_first and scale != 1.0:
    outs = [(v * np.float32(scale)).astype(np.float32) for v in outs]
  return tuple(np.asarray(jnp.asarray(v).astype(out_dtype)) for v in outs)


def assert_legal(actual, *evaluations):
  """Every element of `actual` equals one of the given evaluations (strict, or one multiply fused into the add)."""
  a = bits(actual)
  bad = np.ones(a.shape, bool)
  for e in evaluations:
    bad &= a != bits(e)
  np.testing.assert_equal(int(bad.sum()), 0, err_msg=f"{int(bad.sum())} elements match no legal evaluation")


def make_case(shape, in_dtype, seed):
  b, s, _, h = shape
  kx, ka, kc = jax.random.split(jax.random.PRNGKey(seed), 3)
  inputs = (jax.random.normal(kx, shape, jnp.float32) * 3.0).astype(in_dtype)
  angles = jax.random.uniform(ka, (b, s, h // 2), jnp.float32, 0.0, 2.0e5)  # YaRN angles reach position x freq ~ 1e5
  cos_rows, sin_rows = jnp.cos(angles), jnp.sin(angles)
  cotangent = jax.random.normal(kc, shape, jnp.float32)
  return inputs, cos_rows, sin_rows, cotangent


def make_exact_case(shape, in_dtype, seed):
  """Inputs on which FMA and separate rounding agree: 8 significant bits, magnitudes in [1, 8) for x and the
  cotangent, [0.5, 1) for cos and sin, so every product has at most 16 significant bits and every sum of two of them
  fits in 24. Bit equality on these inputs is then a statement about data movement, sign placement and casts alone."""
  b, s, _, h = shape
  kx, ks, kc, ksc, kcs = jax.random.split(jax.random.PRNGKey(seed), 5)

  def mag(k, shp):  # 8 significant bits, magnitude in [1, 8)
    return jnp.round(jax.random.uniform(k, shp, jnp.float32, 1.0, 8.0) * 16.0) / 16.0

  def sgn(k, shp):
    return jnp.where(jax.random.bernoulli(k, 0.5, shp), -1.0, 1.0)

  inputs = (mag(kx, shape) * sgn(ks, shape)).astype(in_dtype)
  cotangent = mag(kc, shape) * sgn(kcs, shape)
  rows = (b, s, h // 2)
  cos_rows = jnp.round(jax.random.uniform(ksc, rows, jnp.float32, 0.5, 1.0) * 256.0) / 256.0
  sin_rows = jnp.round(jax.random.uniform(kcs, rows, jnp.float32, 0.5, 1.0) * 256.0) / 256.0
  sin_rows = sin_rows * sgn(kx, rows)
  return inputs, cos_rows, sin_rows, cotangent


class RollDirectionTest(unittest.TestCase):
  """`pltpu.roll(x, shift, 1)` must move lane j - shift into lane j (jnp.roll semantics); the swap relies on it."""

  def test_roll_matches_jnp_roll(self):
    x = jnp.arange(8 * 64, dtype=jnp.float32).reshape(8, 64)

    def kernel(x_ref, up_ref, down_ref):
      up_ref[...] = pltpu.roll(x_ref[...], 1, 1)
      down_ref[...] = pltpu.roll(x_ref[...], 63, 1)

    up, down = pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct(x.shape, x.dtype),) * 2, interpret=INTERPRET)(x)
    np.testing.assert_array_equal(np.asarray(up), np.asarray(jnp.roll(x, 1, axis=1)))
    np.testing.assert_array_equal(np.asarray(down), np.asarray(jnp.roll(x, -1, axis=1)))


class PairwiseRopeKernelTest(unittest.TestCase):
  """Kernel (forward and custom_vjp backward) against the reshape form.

  On exactly representable inputs the comparison is bit for bit. On random inputs every element must equal one of
  the two IEEE-legal evaluations of the shared expression (the interpreter compiles the kernel body with XLA:CPU,
  whose LLVM backend fuses the multiply-add; the eager reference does not), so the kernel is shown to compute the
  same products and sums in the same order, which is all a backend-independent test can establish; the on-device
  comparison against the reshape form is in repro/compute/ROPE_KERNEL.md.
  """

  def _new(self, x, cos_rows, sin_rows, out_dtype, scale, **kw):
    """The kernel path (interpreted off TPU)."""
    return rope_pairwise.apply_pairwise_rope(
        x, cos_rows, sin_rows, out_dtype=out_dtype, scale=scale, interpret=INTERPRET, **kw
    )

  def _check_exact(self, shape, in_dtype, out_dtype, seed, scale=1.0, **kw):
    """Bit equality with the reshape form on exactly representable inputs, forward and backward."""
    inputs, cos_rows, sin_rows, cotangent = make_exact_case(shape, in_dtype, seed)
    actual = self._new(inputs, cos_rows, sin_rows, out_dtype, scale, **kw)
    expected = reference_pairwise(inputs, cos_rows, sin_rows, out_dtype, scale)
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertEqual(actual.shape, expected.shape)
    np.testing.assert_array_equal(bits(actual), bits(expected))
    grad_new = jax.grad(
        lambda x: jnp.sum(self._new(x, cos_rows, sin_rows, out_dtype, scale, **kw).astype(jnp.float32) * cotangent)
    )(inputs)
    grad_ref = jax.grad(
        lambda x: jnp.sum(reference_pairwise(x, cos_rows, sin_rows, out_dtype, scale).astype(jnp.float32) * cotangent)
    )(inputs)
    self.assertEqual(grad_new.dtype, grad_ref.dtype)
    np.testing.assert_array_equal(bits(grad_new), bits(grad_ref))

  def _check_random(self, shape, in_dtype, out_dtype, seed, scale=1.0, **kw):
    """Random inputs: every element equals one legal evaluation; the reshape form is the strict one."""
    inputs, cos_rows, sin_rows, cotangent = make_case(shape, in_dtype, seed)
    actual = self._new(inputs, cos_rows, sin_rows, out_dtype, scale, **kw)
    assert_legal(actual, *legal_evaluations(inputs, cos_rows, sin_rows, out_dtype, scale))
    # The reference itself is the strict evaluation (eager, op by op).
    np.testing.assert_array_equal(
        bits(reference_pairwise(inputs, cos_rows, sin_rows, out_dtype, scale)),
        bits(legal_evaluations(inputs, cos_rows, sin_rows, out_dtype, scale)[0]),
    )
    grad_new = jax.grad(
        lambda x: jnp.sum(self._new(x, cos_rows, sin_rows, out_dtype, scale, **kw).astype(jnp.float32) * cotangent)
    )(inputs)
    g = cotangent.astype(out_dtype).astype(jnp.float32) if out_dtype != jnp.float32 else cotangent
    assert_legal(grad_new, *legal_evaluations(g, cos_rows, sin_rows, in_dtype, scale, negate_sin=True, scale_first=True))

  def test_production_q_pe(self):
    # DeepSeek-V3 MLA: d_rope 64, seq 4096, 128 heads, bf16 in and out.
    self._check_exact((1, 4096, 128, 64), jnp.bfloat16, jnp.bfloat16, seed=0)
    self._check_random((1, 4096, 128, 64), jnp.bfloat16, jnp.bfloat16, seed=10)

  def test_production_k_pe_single_head(self):
    self._check_exact((1, 4096, 1, 64), jnp.bfloat16, jnp.bfloat16, seed=1)
    self._check_random((1, 4096, 1, 64), jnp.bfloat16, jnp.bfloat16, seed=11)

  def test_f32_out(self):
    self._check_exact((2, 512, 8, 64), jnp.float32, jnp.float32, seed=2)
    self._check_random((2, 512, 8, 64), jnp.float32, jnp.float32, seed=12)

  def test_scaled_output_d128(self):
    # scale 1.5 keeps the exact case exact (two more significant bits); the random case uses the YaRN factor.
    self._check_exact((2, 256, 4, 128), jnp.bfloat16, jnp.bfloat16, seed=3, scale=1.5)
    self._check_random((2, 256, 4, 128), jnp.bfloat16, jnp.bfloat16, seed=13, scale=0.1 * np.log(40.0) + 1.0)

  def test_small_blocks_and_odd_head_count(self):
    self._check_exact((2, 64, 3, 32), jnp.bfloat16, jnp.bfloat16, seed=4, block_heads=8, block_seq=32, rows_per_iter=16)

  def test_negative_zero_and_infinite_inputs(self):
    # Sign of zero and non-finite values must propagate exactly as in the reshape form (the swap is a select).
    shape = (1, 16, 2, 8)
    x = jnp.zeros(shape, jnp.float32).at[0, 1, 0, 3].set(-0.0).at[0, 2, 1, 4].set(jnp.inf).at[0, 3, 0, 0].set(-jnp.inf)
    _, cos_rows, sin_rows, _ = make_exact_case(shape, jnp.float32, seed=5)
    actual = rope_pairwise.apply_pairwise_rope(x, cos_rows, sin_rows, out_dtype=jnp.float32, interpret=INTERPRET)
    expected = reference_pairwise(x, cos_rows, sin_rows, jnp.float32)
    np.testing.assert_array_equal(bits(actual), bits(expected))


class PairwiseRopeCompileTest(unittest.TestCase):
  """Mosaic compiles the production kernel (forward and backward) for a virtual tpu7x; no hardware needed."""

  def test_compiles_for_tpu7x(self):
    try:
      from jax.experimental import topologies  # pylint: disable=import-outside-toplevel

      topo = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.skipTest(f"virtual tpu7x topology unavailable: {e!r}")
    dev = topo.devices[0]
    x = jax.ShapeDtypeStruct((1, 4096, 128, 64), jnp.bfloat16)
    rows = jax.ShapeDtypeStruct((1, 4096, 32), jnp.float32)
    ct = jax.ShapeDtypeStruct((1, 4096, 128, 64), jnp.bfloat16)

    def fwd(x, cos_rows, sin_rows):
      return rope_pairwise.apply_pairwise_rope(x, cos_rows, sin_rows, out_dtype=jnp.bfloat16)

    def vjp(x, cos_rows, sin_rows, g):
      out, pullback = jax.vjp(fwd, x, cos_rows, sin_rows)
      return out, pullback(g)[0]

    with jax.default_device(dev):
      compiled = jax.jit(vjp).lower(x, rows, rows, ct).compile()
    self.assertIn("pairwise_rope", compiled.as_text())


if __name__ == "__main__":
  unittest.main()
