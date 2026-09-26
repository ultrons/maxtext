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

"""Tests for the ragged gather-reduce wrapper's row-selective TensorCore cast.

The SparseCore kernel writes float32 sums (plus a garbage row) and leaves the slice, the empty-group mask and the
bfloat16 cast to the TensorCore. That pass used to be an XLA fusion streaming the whole float32 buffer; it is now a
Pallas kernel that DMAs only the rows of non-empty groups and writes zeros for the rest, with the same TensorCore
convert on the same float32 values, so the result is bit-identical to ``where(mask, x, 0).astype(bfloat16)``.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.ragged import ragged_gather_reduce_v2 as grv2
import numpy as np
import pytest

# pylint: disable=protected-access


def _interpret_params():
  """The interpreter configuration for a non-TPU backend, or False on a TPU."""
  if jax.default_backend() == "tpu":
    return False
  return pltpu.InterpretParams()


def _f32_rows(rng, num_rows, hidden, specials=True):
  """Random float32 rows; with ``specials``, some rows hold NaN payloads, infinities, signed zeros and denormals."""
  x = (rng.standard_normal((num_rows, hidden)) * 3.0).astype(np.float32)
  if specials:
    bits = x.view(np.uint32)
    rows = rng.choice(num_rows, 8, replace=False)
    bits[rows[0]] = 0x7F800001  # signalling NaN payload
    bits[rows[1]] = 0xFFC00000  # negative quiet NaN
    bits[rows[2]] = 0x7F800000  # +inf
    bits[rows[3]] = 0xFF800000  # -inf
    bits[rows[4]] = 0x80000000  # -0.0
    bits[rows[5]] = 0x00000001  # smallest denormal
    bits[rows[6]] = 0x807FFFFF  # largest negative denormal
    bits[rows[7]] = 0x3F80FFFF  # a tie-adjacent value
    x = bits.view(np.float32)
  return x


def _reference(x, mask, num_rows, hidden):
  """The old wrapper's pass, on the same backend as the kernel under test."""
  out = x[:num_rows, :hidden]
  return jnp.where(mask[:num_rows, None], out, jnp.zeros_like(out)).astype(jnp.bfloat16)


def _bits(a):
  return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint16))


class MaskedRowsToBf16Test(parameterized.TestCase):
  """The Pallas cast against XLA's ``where`` + ``astype`` on the same backend, raw bits compared."""

  @parameterized.named_parameters(
      ("ragged_40pct", 1024, 1024, 0.4, 128, 8),
      ("sparse_5pct", 1024, 1024, 0.05, 128, 8),
      ("dense", 1024, 1024, 1.0, 128, 8),
      ("all_empty", 1024, 1024, 0.0, 128, 8),
      ("small_tiles", 512, 256, 0.4, 32, 4),
      ("unroll_1", 512, 256, 0.4, 64, 1),
      ("production_tile", 512, 7168, 1.0 / 16, 128, 8),
  )
  def test_matches_xla_pass(self, num_rows, hidden, valid_frac, tile_rows, row_unroll):
    rng = np.random.RandomState(0)
    x = jnp.asarray(_f32_rows(rng, num_rows + 1, hidden))  # + the kernel's garbage row
    mask = rng.rand(num_rows) < valid_frac
    if valid_frac not in (0.0, 1.0):
      # tiles with no valid row, one valid row, and every row valid, on top of the random ones
      mask[:tile_rows] = False
      mask[tile_rows : 2 * tile_rows] = False
      mask[tile_rows] = True
      mask[2 * tile_rows : 3 * tile_rows] = True
    mask = jnp.asarray(mask)
    cast = functools.partial(
        grv2._masked_rows_to_bf16,
        num_rows=num_rows,
        hidden_size=hidden,
        out_dtype=jnp.bfloat16,
        tile_rows=tile_rows,
        row_unroll=row_unroll,
        interpret=_interpret_params(),
    )
    got = jax.block_until_ready(cast(x, mask))
    self.assertEqual(got.shape, (num_rows, hidden))
    np.testing.assert_array_equal(_bits(got), _bits(_reference(x, mask, num_rows, hidden)))
    # the garbage row is never read: rows past num_rows do not influence the output
    got2 = jax.block_until_ready(cast(x.at[num_rows].set(jnp.nan), mask))
    np.testing.assert_array_equal(_bits(got2), _bits(got))

  def test_odd_counts_per_tile(self):
    # Every tile has a valid-row count that is not a multiple of the unroll, so the padded slots repeat rows.
    num_rows, hidden, tile_rows, row_unroll = 256, 128, 64, 8
    rng = np.random.RandomState(1)
    x = jnp.asarray(_f32_rows(rng, num_rows + 1, hidden, specials=False))
    mask = np.zeros(num_rows, bool)
    for t, count in enumerate((1, 7, 9, 63)):
      mask[t * tile_rows : t * tile_rows + count] = True
    mask = jnp.asarray(mask)
    got = grv2._masked_rows_to_bf16(
        x, mask, num_rows, hidden, jnp.bfloat16, tile_rows=tile_rows, row_unroll=row_unroll, interpret=_interpret_params()
    )
    np.testing.assert_array_equal(_bits(got), _bits(_reference(x, mask, num_rows, hidden)))


class WrapperTest(absltest.TestCase):

  def test_cpu_fallback_contract(self):
    if jax.default_backend() == "tpu":
      self.skipTest("CPU fallback path")
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.standard_normal((512, 256)).astype(np.float32)).astype(jnp.bfloat16)
    indices = jnp.asarray(rng.randint(0, 512, 4096).astype(np.int32))
    w = jnp.asarray(rng.standard_normal(4096).astype(np.float32))
    valid = jnp.asarray(rng.rand(4096) < 0.3)
    out = grv2.ragged_gather_reduce(x, indices, w, valid, reduce_group_size=8)
    self.assertEqual(out.shape, (512, 256))
    self.assertEqual(out.dtype, jnp.bfloat16)
    masked = ~np.asarray(valid).reshape(-1, 8).any(-1)
    self.assertTrue(np.all(np.asarray(out, np.float32)[masked] == 0))


@pytest.mark.tpu_only
class SparseCoreKernelTest(parameterized.TestCase):
  """The full wrapper on a TPU with SparseCore against the old two-pass result."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("needs a TPU")
    if pltpu.get_tpu_info().sparse_core is None:
      self.skipTest("needs a TPU with SparseCore")

  @parameterized.named_parameters(
      ("bf16_ragged", 512, 4096, 4096, 8, 0.3),
      ("bf16_sparse", 512, 4096, 4096, 8, 0.05),
      ("bf16_dense", 512, 4096, 4096, 8, 1.0),
      ("bf16_all_invalid", 512, 4096, 4096, 8, 0.0),
      ("bf16_mid", 8192, 4096, 65536, 8, 0.1),
      ("bf16_production_1of16", 40960, 7168, 262144, 8, 1.0 / 16),
  )
  def test_matches_sequential_reference(self, rows, hidden, n, group, valid_frac):
    rng = np.random.RandomState(0)
    x_np = rng.standard_normal((rows, hidden)).astype(np.float32)
    x = jnp.asarray(x_np).astype(jnp.bfloat16)
    indices = rng.randint(0, rows, n).astype(np.int32)
    w = rng.standard_normal(n).astype(np.float32)
    valid = rng.rand(n) < valid_frac
    out = jax.jit(functools.partial(grv2.ragged_gather_reduce, reduce_group_size=group))(
        x, jnp.asarray(indices), jnp.asarray(w), jnp.asarray(valid)
    )
    out = jax.block_until_ready(out)
    # sequential f32 sum of the valid rows of each group (the kernel's order), rounded by this device's convert
    xb = np.asarray(x.astype(jnp.float32))
    ref = np.zeros((n // group, hidden), np.float32)
    for g in range(n // group):
      acc = None
      for r in range(g * group, (g + 1) * group):
        if valid[r]:
          val = (xb[indices[r]] * np.float32(w[r])).astype(np.float32)
          acc = val if acc is None else (acc + val).astype(np.float32)
      if acc is not None:
        ref[g] = acc
    np.testing.assert_array_equal(_bits(out), _bits(jnp.asarray(ref).astype(jnp.bfloat16)))


if __name__ == "__main__":
  absltest.main()
