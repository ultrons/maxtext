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

"""Pairwise (interleaved) rotary embedding as a Pallas TPU kernel.

On the last axis of `x` every lane pair (2k, 2k + 1) is rotated by the angle of frequency k:

  out[2k]     = x[2k]     * cos_k - x[2k + 1] * sin_k
  out[2k + 1] = x[2k + 1] * cos_k + x[2k]     * sin_k

computed in f32 with exactly the per-element operation order of the reshape form in
`YarnRotaryEmbedding` (`x * cos + swapped * (sin * sign)`, where `swapped` exchanges the two lanes of each pair
and `sign` is -1 on even lanes and +1 on odd lanes), so the results are bit-identical to that form. The pair swap
is two lane rotations and a select inside VMEM; the activation is read once and written once in the
[batch, heads, seq, head_dim] order its producer already uses, so no relayout copy is needed around the rotation.

The backward pass is the same kernel: the swap is its own transpose and `swap(sin * sign) = -(sin * sign)`, so
d_x = g * cos + swap(g) * (-(sin * sign)).
"""

import dataclasses
import functools
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@dataclasses.dataclass(frozen=True)
class RopeKernelParams:
  """Static parameters of one kernel application."""

  out_dtype: Any
  in_dtype: Any  # dtype of the rotated input; the backward emits its cotangent in this dtype
  scale: float = 1.0  # multiplies the rotated value before the output cast (the forward); the backward applies it first
  block_heads: int = 8
  block_seq: int = 1024
  rows_per_iter: int = 256
  interpret: bool = False


def _rope_kernel(x_ref, cos_ref, ss_ref, o_ref, *, rows_per_iter, scale_in, scale_out):
  """One (block_heads, block_seq, head_dim) tile.

  x_ref: (bn, bs, h) input; cos_ref / ss_ref: (bs, h) f32 with the per-pair value on both lanes of the pair
  (ss = sin * sign); o_ref: (bn, bs, h) output.
  """
  bn, bs, h = x_ref.shape
  n_iter = bs // rows_per_iter

  def body(i, carry):
    start = pl.multiple_of(i * rows_per_iter, rows_per_iter)
    rows = pl.ds(start, rows_per_iter)
    cos = cos_ref[rows, :]
    ss = ss_ref[rows, :]
    lane = lax.broadcasted_iota(jnp.int32, (rows_per_iter, h), 1)
    even = (lane % 2) == 0
    for j in range(bn):
      x = x_ref[j, rows, :].astype(jnp.float32)
      if scale_in != 1.0:
        x = x * scale_in
      # Lane j takes x[j + 1] on even lanes and x[j - 1] on odd lanes: the pair swap.
      swapped = jnp.where(even, pltpu.roll(x, h - 1, 1), pltpu.roll(x, 1, 1))
      out = x * cos + swapped * ss
      if scale_out != 1.0:
        out = out * scale_out
      o_ref[j, rows, :] = out.astype(o_ref.dtype)
    return carry

  lax.fori_loop(0, n_iter, body, 0)


def kernel_supports(shape) -> bool:
  """Whether `apply_pairwise_rope` can tile an input of this [B, S, N, H] shape (else callers use the reshape form)."""
  if len(shape) != 4:
    return False
  _, s, _, h = shape
  return h % 2 == 0 and h <= 128 and s % 8 == 0


def pairwise_rope_bnsh(
    x,
    cos,
    ss,
    *,
    out_dtype,
    scale_in=1.0,
    scale_out=1.0,
    block_heads=8,
    block_seq=1024,
    rows_per_iter=256,
    interpret=False,
):
  """Applies the rotation to `x` of shape [B, N, S, H] with `cos`, `ss` of shape [B, S, H] (f32)."""
  b, n, s, h = x.shape
  if h % 2 or h > 128:
    raise ValueError(f"head_dim must be even and at most 128 lanes, got {h}")
  if cos.shape != (b, s, h) or ss.shape != (b, s, h):
    raise ValueError(f"cos/ss must be [B, S, H] = {(b, s, h)}, got {cos.shape} / {ss.shape}")
  bn = min(block_heads, n)
  bs = min(block_seq, s)
  rows = min(rows_per_iter, bs)
  if n % bn or s % bs or bs % rows or rows % 8:
    raise ValueError(f"heads {n} / seq {s} must be multiples of the block sizes ({bn}, {bs}, rows {rows})")
  kernel = functools.partial(_rope_kernel, rows_per_iter=rows, scale_in=float(scale_in), scale_out=float(scale_out))
  x_spec = pl.BlockSpec((None, bn, bs, h), lambda bi, ni, si: (bi, ni, si, 0))
  row_spec = pl.BlockSpec((None, bs, h), lambda bi, ni, si: (bi, si, 0))
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((b, n, s, h), out_dtype),
      grid=(b, n // bn, s // bs),
      in_specs=[x_spec, row_spec, row_spec],
      out_specs=x_spec,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel"),
          vmem_limit_bytes=48 * 1024 * 1024,
      ),
      interpret=interpret,
      name=f"pairwise_rope-b{b}-n{n}-s{s}-h{h}",
  )(x, cos, ss)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def pairwise_rope(x, cos, ss, params: RopeKernelParams):
  """x: [B, N, S, H]; cos, ss = sin * sign: [B, S, H] f32; returns the rotated x in `params.out_dtype`."""
  return _pairwise_rope_fwd_impl(x, cos, ss, params)


def _pairwise_rope_fwd_impl(x, cos, ss, params):
  return pairwise_rope_bnsh(
      x,
      cos,
      ss,
      out_dtype=params.out_dtype,
      scale_out=params.scale,
      block_heads=params.block_heads,
      block_seq=params.block_seq,
      rows_per_iter=params.rows_per_iter,
      interpret=params.interpret,
  )


def _pairwise_rope_fwd(x, cos, ss, params):
  """custom_vjp forward: the kernel output and the (small) cos / sin rows as residuals."""
  return _pairwise_rope_fwd_impl(x, cos, ss, params), (cos, ss)


def _pairwise_rope_bwd(params, residuals, g):
  """custom_vjp backward: the same kernel on the cotangent with sin * sign negated; zero cotangents for the rows."""
  cos, ss = residuals
  dx = pairwise_rope_bnsh(
      g,
      cos,
      -ss,
      out_dtype=params.in_dtype,
      scale_in=params.scale,
      block_heads=params.block_heads,
      block_seq=params.block_seq,
      rows_per_iter=params.rows_per_iter,
      interpret=params.interpret,
  )
  return dx, jnp.zeros_like(cos), jnp.zeros_like(ss)


pairwise_rope.defvjp(_pairwise_rope_fwd, _pairwise_rope_bwd)


def pairwise_rope_reshape_form(inputs, cos_rows, sin_rows, *, out_dtype, scale=1.0):
  """The reshape form of the same rotation (used for shapes the kernel does not tile; same arithmetic)."""
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


def apply_pairwise_rope(
    inputs, cos_rows, sin_rows, *, out_dtype, scale=1.0, interpret=False, block_heads=8, block_seq=1024, rows_per_iter=256
):
  """Pairwise RoPE on `inputs` of shape [B, S, N, H] with per-pair `cos_rows`, `sin_rows` of shape [B, S, H // 2].

  Equivalent to the reshape form: `pairs = inputs.reshape(B, S, N, H // 2, 2).astype(f32)`,
  `(pairs * cos + flip(pairs) * sin * [-1, 1]).reshape(B, S, N, H) * scale`, cast to `out_dtype`.
  """
  h = inputs.shape[-1]
  cos = jnp.repeat(cos_rows.astype(jnp.float32), 2, axis=-1)  # [B, S, H]: cos_k on lanes 2k and 2k + 1
  sin = jnp.repeat(sin_rows.astype(jnp.float32), 2, axis=-1)
  sign = jnp.tile(jnp.asarray([-1.0, 1.0], dtype=jnp.float32), h // 2)
  ss = sin * sign  # exact: a multiply by +-1
  params = RopeKernelParams(
      out_dtype=jnp.dtype(out_dtype),
      in_dtype=jnp.dtype(inputs.dtype),
      scale=float(scale),
      block_heads=block_heads,
      block_seq=block_seq,
      rows_per_iter=rows_per_iter,
      interpret=interpret,
  )
  # [B, S, N, H] -> [B, N, S, H]: a bitcast when the producer already holds the head-major layout XLA gives the
  # attention inputs; the kernel tiles (heads, seq, head_dim).
  x = jnp.transpose(inputs, (0, 2, 1, 3))
  out = pairwise_rope(x, cos, ss, params)
  return jnp.transpose(out, (0, 2, 1, 3))
