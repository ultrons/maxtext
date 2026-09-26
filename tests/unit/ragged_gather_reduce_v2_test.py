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

"""Tests for the SparseCore ragged gather-reduce kernel's in-kernel rounding, packing and zero-fill.

The kernel used to write a float32 output with a garbage row and leave the slice, empty-group mask and bfloat16 cast
to a TensorCore pass. It now writes the final bfloat16 rows itself (two output rows packed per uint32 word, as the TPU
bfloat16 layout stores them) and zero-fills the words of empty groups. These tests pin the contract that the result is
bit-identical to the old path: same float32 accumulation order, round to nearest even, and the TensorCore convert's
treatment of NaN and denormals.

The SparseCore kernel cannot run under the Pallas CPU interpreter (SparseCore meshes, indirect DMAs and
``plsc.bitcast`` have no interpreter rules), so on CPU the kernel's data flow is emulated in numpy with the kernel's
own plan/reduce/round/pack functions and compared against an independent reference of the old semantics; the real
kernel is checked against the same reference on a TPU with SparseCore.
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

# f32 bit pattern -> bf16 bit pattern of XLA's f32->bf16 convert, measured on a TPU (v5p, jax 0.11.0.dev20260630,
# ``scratchpad/cvt_probe.py``): RNE, denormals flushed to a signed zero, every NaN to 0x7FC0 (sign dropped). XLA:CPU
# differs on two of these (it keeps denormals and the NaN sign), which is why the special cases are pinned here.
_TPU_CONVERT_TABLE = (
    (0x7FC00000, 0x7FC0),
    (0x7F800001, 0x7FC0),
    (0x7FFFFFFF, 0x7FC0),
    (0xFF800001, 0x7FC0),
    (0x7F80FFFF, 0x7FC0),
    (0xFFC00000, 0x7FC0),
    (0x7FBFFFFF, 0x7FC0),
    (0x00000001, 0x0000),
    (0x00008000, 0x0000),
    (0x007FFFFF, 0x0000),
    (0x0000FFFF, 0x0000),
    (0x80008000, 0x8000),
    (0x3F808000, 0x3F80),
    (0x3F818000, 0x3F82),
    (0x3F80FFFF, 0x3F81),
    (0x3F807FFF, 0x3F80),
    (0x7F7FFFFF, 0x7F80),
    (0xFF7FFFFF, 0xFF80),
    (0x80000000, 0x8000),
    (0x00000000, 0x0000),
    (0x7F800000, 0x7F80),
    (0xFF800000, 0xFF80),
)


def _f32_to_bf16_bits_reference(values: np.ndarray) -> np.ndarray:
  """Independent numpy model of the TensorCore f32->bf16 convert (see ``_TPU_CONVERT_TABLE``)."""
  bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
  lsb = (bits >> 16) & 1
  rounded = ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF
  sign_only = (bits >> 16) & 0x8000
  is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
  is_denormal = (bits & 0x7F800000) == 0
  return np.where(is_nan, 0x7FC0, np.where(is_denormal, sign_only, rounded)).astype(np.uint16)


def _old_path_reference(x, indices, topk_weights, valid_rows_mask, reduce_group_size):
  """The old wrapper's result: sequential f32 sum of the valid rows of each group, masked, then the TC convert.

  Returns the output as raw bits (uint16 for bfloat16 inputs, uint32 for float32 inputs).
  """
  x = np.asarray(x.astype(jnp.float32))
  indices = np.asarray(indices)
  w = np.asarray(topk_weights.astype(jnp.float32))
  valid = np.asarray(valid_rows_mask)
  num_groups = indices.shape[0] // reduce_group_size
  out = np.zeros((num_groups, x.shape[1]), np.float32)
  for group in range(num_groups):
    acc = None
    for r in range(group * reduce_group_size, (group + 1) * reduce_group_size):
      if not valid[r]:
        continue
      val = (x[indices[r]] * np.float32(w[r])).astype(np.float32)
      acc = val if acc is None else (acc + val).astype(np.float32)
    if acc is not None:
      out[group] = acc
  return out


def _emulate_kernel(x, indices, topk_weights, valid_rows_mask, reduce_group_size, *, num_row_partitions, num_simd_lanes):
  """Numpy emulation of the kernel's data flow, using the kernel's own bookkeeping and arithmetic helpers.

  Mirrors one column partition of the SparseCore kernel: per row partition, the validity-sorted rows are walked in
  blocks of ``row_chunk_size`` and sub-chunks of ``num_simd_lanes`` rows; the segmented reduction, rounding and word
  packing run through ``_reduce_row``, ``_round_f32_bits_to_bf16`` and ``_pack_output_word`` with the same carries
  the kernel keeps across sub-chunks and blocks; every lane's scatter from ``_plan_row_block`` is applied in order
  (partial words included, duplicates included) onto an HBM image pre-filled with garbage, and the empty words from
  ``_empty_output_words`` are zero-filled first. Returns the output rows as raw bits.
  """
  out_dtype = x.dtype
  out_pack = 32 // jax.dtypes.itemsize_bits(out_dtype)
  input_size = indices.shape[0]
  hidden = x.shape[1]
  num_row_subchunks, row_chunk_size = grv2._calculate_row_tiling(input_size, num_simd_lanes, num_row_partitions)
  padded_input_size = grv2._align_to(input_size, num_row_partitions * reduce_group_size * out_pack)
  valid_padded = jnp.pad(valid_rows_mask, (0, padded_input_size - input_size), constant_values=False)
  sorted_by_validity, num_rows_per_partition, mask = grv2._preprocess(
      valid_padded, reduce_group_size, num_row_partitions, num_simd_lanes, row_chunk_size
  )
  empty_words, num_empty_words = grv2._empty_output_words(mask, num_row_partitions, out_pack, num_simd_lanes)
  sorted_by_validity = np.asarray(sorted_by_validity)
  num_rows_per_partition = np.asarray(num_rows_per_partition)
  empty_words = np.asarray(empty_words)
  num_empty_words = np.asarray(num_empty_words)

  x_f32 = np.asarray(x.astype(jnp.float32))
  indices = np.asarray(indices)
  w = np.asarray(topk_weights.astype(jnp.float32))
  num_words = padded_input_size // reduce_group_size // out_pack
  # Garbage-filled HBM image of the output words: everything must be written or zero-filled.
  hbm = np.full((num_words, hidden), 0xDEADBEEF, np.uint32)

  rows_per_partition_padded = sorted_by_validity.shape[0] // num_row_partitions
  words_per_partition_padded = empty_words.shape[0] // num_row_partitions
  for partition in range(num_row_partitions):
    # Zero-fill of the empty words (whole blocks; the padded slots repeat an empty word).
    n_empty = int(num_empty_words[partition])
    plist = empty_words[partition * words_per_partition_padded : (partition + 1) * words_per_partition_padded]
    for block in range(-(-n_empty // num_simd_lanes)):
      for i in range(num_simd_lanes):
        hbm[plist[block * num_simd_lanes + i]] = 0

    num_rows = int(num_rows_per_partition[partition])
    order = sorted_by_validity[partition * rows_per_partition_padded : (partition + 1) * rows_per_partition_padded]
    prev_dst_smem = -out_pack
    carry_acc = np.full((num_row_subchunks * 0 + 1, hidden), np.nan, np.float32)[0]  # uninitialised VMEM
    carry_word = np.full((hidden,), 0xBAADF00D, np.uint32)
    num_row_blocks = -(-num_rows // row_chunk_size)
    for block in range(num_row_blocks):
      base = block * row_chunk_size
      dst_rows = [int(order[base + r]) // reduce_group_size for r in range(row_chunk_size)]
      row_valid = [base + r < num_rows for r in range(row_chunk_size)]
      dst_eff, is_pad, dma_src, dma_dst = grv2._plan_row_block(
          dst_rows,
          row_valid,
          prev_dst_smem,
          num_simd_lanes=num_simd_lanes,
          num_row_subchunks=num_row_subchunks,
          out_pack=out_pack,
      )
      dst_eff = [int(d) for d in dst_eff]
      is_pad = [int(p) for p in is_pad]
      for s in range(num_row_subchunks):
        out_vmem = np.zeros((num_simd_lanes, hidden), np.uint32)
        prev_acc = carry_acc
        prev_word = carry_word
        for i in range(num_simd_lanes):
          r = s * num_simd_lanes + i
          src_row = int(order[base + r])
          data = (x_f32[indices[src_row]] * np.float32(w[src_row])).astype(np.float32)
          dst = dst_eff[r]
          prev_dst = prev_dst_smem if r == 0 else dst_eff[r - 1]
          acc = np.asarray(grv2._reduce_row(jnp.asarray(prev_acc), jnp.asarray(data), dst == prev_dst, is_pad[r] != 0))
          if out_pack == 1:
            word = acc.view(np.uint32)
          else:
            bf16_bits = grv2._round_f32_bits_to_bf16(jnp.asarray(acc.view(np.uint32)))
            word = np.asarray(
                grv2._pack_output_word(jnp.asarray(prev_word), bf16_bits, dst, dst // out_pack == prev_dst // out_pack)
            )
          out_vmem[i] = word
          prev_acc, prev_word = acc, word
        carry_acc, carry_word = prev_acc, prev_word
        for i in range(num_simd_lanes):
          hbm[int(dma_dst[s][i])] = out_vmem[int(dma_src[s][i])]
      prev_dst_smem = dst_eff[-1]

  if out_pack == 1:
    return hbm[: input_size // reduce_group_size]
  # Unpack the row pairs: even row in the low half.
  rows = np.empty((num_words * 2, hidden), np.uint16)
  rows[0::2] = (hbm & 0xFFFF).astype(np.uint16)
  rows[1::2] = (hbm >> 16).astype(np.uint16)
  return rows[: input_size // reduce_group_size]


def _make_case(seed, num_x_rows, hidden, n, valid_frac, dtype, nan_rows=0, negzero=False, valid_prefix=False):
  """Random inputs: x rows (optionally with NaN rows or many signed zeros), gather indices, f32 weights, validity."""
  rng = np.random.RandomState(seed)
  x = rng.standard_normal((num_x_rows, hidden)).astype(np.float32)
  if negzero:
    x[rng.choice(num_x_rows, num_x_rows // 4, replace=False)] = -0.0
    x[rng.choice(num_x_rows, num_x_rows // 8, replace=False)] = 0.0
  if nan_rows:
    x[rng.choice(num_x_rows, nan_rows, replace=False)] = np.nan
  x = jnp.asarray(x).astype(dtype)
  indices = jnp.asarray(rng.randint(0, num_x_rows, n).astype(np.int32))
  w = jnp.asarray(rng.standard_normal(n).astype(np.float32))
  if valid_prefix:
    valid = np.arange(n) < int(n * valid_frac)
  else:
    valid = rng.rand(n) < valid_frac
  return x, indices, w, jnp.asarray(valid)


def _reference_bits(x, indices, w, valid, group):
  ref = _old_path_reference(x, indices, w, valid, group)
  if x.dtype == jnp.float32:
    return ref.view(np.uint32)
  return _f32_to_bf16_bits_reference(ref)


class RoundingAndPackingTest(parameterized.TestCase):
  """The in-kernel f32->bf16 rounding and the row-pair word packing."""

  def test_round_matches_measured_tensorcore_convert(self):
    # The SparseCore's float32 arithmetic flushes denormals (measured on v5p: sums of products down to 2^-136 come
    # out as signed zeros from the old f32 kernel and the new one alike), so the kernel helper only has to match the
    # TensorCore convert on non-denormal inputs; the denormal rows of the table are covered by the numpy model below.
    src = np.array([a for a, _ in _TPU_CONVERT_TABLE], np.uint32)
    expected = np.array([b for _, b in _TPU_CONVERT_TABLE], np.uint32)
    not_denormal = (src & 0x7F800000) != 0
    got = np.asarray(grv2._round_f32_bits_to_bf16(jnp.asarray(src)))
    np.testing.assert_array_equal(got[not_denormal], expected[not_denormal])
    np.testing.assert_array_equal(_f32_to_bf16_bits_reference(src.view(np.float32)).astype(np.uint32), expected)
    # The kernel passes the NaN mask from a float compare; same result.
    is_nan = jnp.asarray(src.view(np.float32)) != jnp.asarray(src.view(np.float32))
    got2 = np.asarray(grv2._round_f32_bits_to_bf16(jnp.asarray(src), is_nan))
    np.testing.assert_array_equal(got2[not_denormal], expected[not_denormal])

  def test_round_matches_xla_convert_on_normal_values(self):
    # XLA:CPU and the TensorCore agree on everything except NaN sign and denormals; check RNE on a million normals,
    # including exact ties, both signs, and the overflow-to-inf edge.
    rng = np.random.RandomState(0)
    vals = np.concatenate(
        [
            rng.standard_normal(1 << 20).astype(np.float32) * np.float32(1e3),
            np.float32(1.0) + np.arange(4096, dtype=np.float32) * np.float32(2**-16),
            np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max, 3.0e38, -3.0e38, 65504.0], np.float32),
        ]
    )
    xla = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(vals).astype(jnp.bfloat16), jnp.uint16)).astype(np.uint32)
    ours = np.asarray(grv2._round_f32_bits_to_bf16(jnp.asarray(vals.view(np.uint32))))
    np.testing.assert_array_equal(ours, xla)
    np.testing.assert_array_equal(ours, _f32_to_bf16_bits_reference(vals).astype(np.uint32))

  def test_pack_output_word(self):
    prev = jnp.asarray(np.array([0x11112222, 0x33334444], np.uint32))
    bits = jnp.asarray(np.array([0xABCD, 0xABCD], np.uint32))
    # Same word: the even row replaces the low half, the odd row the high half.
    np.testing.assert_array_equal(np.asarray(grv2._pack_output_word(prev, bits, 4, True)), [0x1111ABCD, 0x3333ABCD])
    np.testing.assert_array_equal(np.asarray(grv2._pack_output_word(prev, bits, 5, True)), [0xABCD2222, 0xABCD4444])
    # New word: the other half starts at zero (an empty partner row).
    np.testing.assert_array_equal(np.asarray(grv2._pack_output_word(prev, bits, 4, False)), [0x0000ABCD, 0x0000ABCD])
    np.testing.assert_array_equal(np.asarray(grv2._pack_output_word(prev, bits, 5, False)), [0xABCD0000, 0xABCD0000])

  def test_reduce_row_keeps_negative_zero_through_padding(self):
    prev = jnp.asarray(np.array([-0.0, 1.5], np.float32))
    data = jnp.asarray(np.array([0.0, 2.0], np.float32))
    padded = np.asarray(grv2._reduce_row(prev, data, True, True))
    np.testing.assert_array_equal(padded.view(np.uint32), np.asarray(prev).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(grv2._reduce_row(prev, data, True, False)), [0.0, 3.5])
    np.testing.assert_array_equal(np.asarray(grv2._reduce_row(prev, data, False, False)), [0.0, 2.0])


class EmptyWordListTest(parameterized.TestCase):

  @parameterized.parameters((0.0,), (0.05,), (0.5,), (0.97,), (1.0,))
  def test_lists_exactly_the_empty_words(self, valid_frac):
    rng = np.random.RandomState(1)
    num_row_partitions, out_pack, lanes = 4, 2, 16
    num_groups = 4 * 8 * 2 * 37  # not a multiple of lanes per partition
    mask = jnp.asarray(rng.rand(num_groups) < valid_frac)
    empty_words, num_empty = grv2._empty_output_words(mask, num_row_partitions, out_pack, lanes)
    empty_words, num_empty = np.asarray(empty_words), np.asarray(num_empty)
    word_valid = np.asarray(mask).reshape(-1, out_pack).any(-1)
    words_per_partition = word_valid.shape[0] // num_row_partitions
    pad_to = -(-words_per_partition // lanes) * lanes
    self.assertEqual(empty_words.shape, (num_row_partitions * pad_to,))
    self.assertEqual(num_empty.shape, (lanes,))
    for p in range(num_row_partitions):
      expected = (
          np.nonzero(~word_valid[p * words_per_partition : (p + 1) * words_per_partition])[0] + p * words_per_partition
      )
      got = empty_words[p * pad_to : (p + 1) * pad_to]
      count = int(num_empty[p])
      self.assertEqual(count, expected.size)
      np.testing.assert_array_equal(got[:count], expected)
      if count:
        # Padded slots re-zero the partition's first empty word, never a valid one.
        np.testing.assert_array_equal(got[count:], expected[0])


class KernelDataFlowTest(parameterized.TestCase):
  """The kernel's algorithm (continuation rows, pair packing, carries, zero-fill) reproduces the old path bit for bit."""

  @parameterized.named_parameters(
      ("bf16_ragged", 512, 256, 4096, 8, 0.3, jnp.bfloat16, {}),
      ("bf16_sparse_many_empty_pairs", 512, 256, 4096, 8, 0.05, jnp.bfloat16, {}),
      ("bf16_dense_no_empty_word", 512, 256, 4096, 8, 1.0, jnp.bfloat16, {}),
      ("bf16_all_invalid", 512, 256, 4096, 8, 0.0, jnp.bfloat16, {}),
      ("bf16_padding_rows", 512, 256, 4096 + 24, 8, 0.3, jnp.bfloat16, {}),
      ("bf16_nan_rows", 512, 256, 4096, 8, 0.3, jnp.bfloat16, {"nan_rows": 40}),
      ("bf16_negative_zero", 512, 256, 4096, 8, 0.3, jnp.bfloat16, {"negzero": True}),
      ("bf16_group1_prefix", 1024, 128, 1024, 1, 0.6, jnp.bfloat16, {"valid_prefix": True}),
      ("f32_ragged", 512, 256, 4096, 8, 0.3, jnp.float32, {}),
      ("bf16_one_partition_row_only", 512, 256, 4096, 8, 1.0 / 4096, jnp.bfloat16, {}),
  )
  def test_emulated_kernel_matches_old_path(self, rows, hidden, n, group, valid_frac, dtype, kwargs):
    x, indices, w, valid = _make_case(0, rows, hidden, n, valid_frac, dtype, **kwargs)
    expected = _reference_bits(x, indices, w, valid, group)
    for num_row_partitions, lanes in ((4, 16), (8, 8), (1, 16)):
      got = _emulate_kernel(x, indices, w, valid, group, num_row_partitions=num_row_partitions, num_simd_lanes=lanes)
      np.testing.assert_array_equal(got, expected, err_msg=f"{num_row_partitions=} {lanes=}")

  def test_padding_rows_are_covered_by_the_reference(self):
    # Sanity check of the harness: a case whose partitions end mid-block has padding rows in every partition.
    x, indices, w, valid = _make_case(3, 256, 128, 8 * 4 * 2 * 13, 0.5, jnp.bfloat16)
    num_rows, row_chunk = grv2._calculate_row_tiling(indices.shape[0], 16, 4)
    self.assertGreater(num_rows, 1)
    self.assertNotEqual((indices.shape[0] // 4) % row_chunk, 0)
    got = _emulate_kernel(x, indices, w, valid, 8, num_row_partitions=4, num_simd_lanes=16)
    np.testing.assert_array_equal(got, _reference_bits(x, indices, w, valid, 8))


class WrapperTest(absltest.TestCase):

  def test_cpu_fallback_contract(self):
    if jax.default_backend() == "tpu":
      self.skipTest("CPU fallback path")
    x, indices, w, valid = _make_case(0, 512, 256, 4096, 0.3, jnp.bfloat16)
    out = grv2.ragged_gather_reduce(x, indices, w, valid, reduce_group_size=8)
    self.assertEqual(out.shape, (512, 256))
    self.assertEqual(out.dtype, jnp.bfloat16)
    masked = ~np.asarray(valid).reshape(-1, 8).any(-1)
    self.assertTrue(np.all(np.asarray(out, np.float32)[masked] == 0))

  def test_cost_estimate_counts_output_in_input_dtype(self):
    est = grv2.get_cost_estimate(
        padded_input_size=4096, aligned_hidden_size=256, reduce_group_size=8, input_dtype_bytes=2
    )
    self.assertEqual(est.bytes_accessed, 4096 * 256 * 2 + 3 * 4096 * 4 + 512 * 256 * 2)


@pytest.mark.tpu_only
class SparseCoreKernelTest(parameterized.TestCase):
  """The real kernel on a TPU with SparseCore against the sequential reference of the old path."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("needs a TPU")
    if pltpu.get_tpu_info().sparse_core is None:
      self.skipTest("needs a TPU with SparseCore")

  # hidden >= 4096 keeps the kernel's own partitioning legal (num_row_partitions <= num_simd_lanes) on v5p and v7x;
  # narrower inputs trip the same assertion in the unchanged configuration code.
  @parameterized.named_parameters(
      ("bf16_ragged", 512, 4096, 4096, 8, 0.3, jnp.bfloat16, {}),
      ("bf16_sparse", 512, 4096, 4096, 8, 0.05, jnp.bfloat16, {}),
      ("bf16_dense", 512, 4096, 4096, 8, 1.0, jnp.bfloat16, {}),
      ("bf16_all_invalid", 512, 4096, 4096, 8, 0.0, jnp.bfloat16, {}),
      ("bf16_padding_rows", 512, 4096, 4096 + 24, 8, 0.3, jnp.bfloat16, {}),
      ("bf16_nan_rows", 512, 4096, 4096, 8, 0.3, jnp.bfloat16, {"nan_rows": 40}),
      ("bf16_negative_zero", 512, 4096, 4096, 8, 0.3, jnp.bfloat16, {"negzero": True}),
      ("bf16_group1_prefix", 1024, 4096, 1024, 1, 0.6, jnp.bfloat16, {"valid_prefix": True}),
      ("f32_ragged", 1024, 4096, 8192, 8, 0.3, jnp.float32, {}),
      ("bf16_mid", 8192, 4096, 65536, 8, 0.1, jnp.bfloat16, {}),
  )
  def test_kernel_matches_old_path(self, rows, hidden, n, group, valid_frac, dtype, kwargs):
    x, indices, w, valid = _make_case(0, rows, hidden, n, valid_frac, dtype, **kwargs)
    out = jax.jit(functools.partial(grv2.ragged_gather_reduce, reduce_group_size=group))(x, indices, w, valid)
    out = jax.block_until_ready(out)
    self.assertEqual(out.dtype, dtype)
    bits_dtype = jnp.uint32 if dtype == jnp.float32 else jnp.uint16
    got = np.asarray(jax.lax.bitcast_convert_type(out, bits_dtype))
    np.testing.assert_array_equal(got, _reference_bits(x, indices, w, valid, group))


if __name__ == "__main__":
  absltest.main()
