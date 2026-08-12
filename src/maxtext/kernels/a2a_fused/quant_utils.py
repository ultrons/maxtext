# quant_utils.py — PHASE E v0: home-side fp8 (e4m3) quantizers for the fused
# a2a MoE layer (PHASE_E_SPEC.md deliverable 3).
#
# quantize_rows:    per-token (per-row) amax quantization of the dispatch lhs.
#                   The scale rides the wire as the sidecar (K1's fp8 path);
#                   spec: scale = amax/448 guard-banded, zero-amax -> scale 1.
# quantize_weights: per-block (>= 256 along K) weight quantization producing
#                   gmm_v2's rhs_scale contract [G, num_blocks, 1, N].
#                   Neither gather/bench_gmm_v2.py (bf16-only) nor the fork
#                   ships a weight quantizer — this is the reference math the
#                   validated fp8-gmm bench session used (per-block amax along
#                   K, f32 scales), written once here.
#
# fp8-MXU cliff rule (MEASURED, opcode-verified in the fp8-gmm session):
# quant_block_size < mxu_column_size (256 on v7x) takes the dequantize-BEFORE-
# matmul path -> the matmul silently runs bf16 -> ZERO fp8 win. quantize_weights
# asserts block_k >= 256.
import jax
import jax.numpy as jnp

E4M3_MAX = 448.0  # jnp.finfo(jnp.float8_e4m3fn).max
MIN_QUANT_BLOCK = 256  # = v7x mxu_column_size; the dequant-before-bf16 cliff


def _guard_scale(scale: jax.Array) -> jax.Array:
  """Zero/underflow guard: scale <= 0 (zero-amax rows, or amax/448 underflow
  to 0 in f32) would make x/scale -> inf/NaN and propagate. Spec: scale 1."""
  return jnp.where(scale > 0, scale, jnp.float32(1.0))


def quantize_rows(x: jax.Array):
  """Per-token amax quantize: x [M, K] float -> (e4m3 [M, K], f32 scale [M]).

  dequant(q, s)[m, :] = q[m, :].astype(f32) * s[m]. The max |element| of each
  row maps to exactly +-448 (e4m3-exact), so no overflow guard-band beyond the
  f32 rounding of amax/448 is needed (448*(1+2^-24) still rounds to 448; the
  e4m3 overflow threshold is 464).
  """
  assert x.ndim == 2, x.shape
  xf = x.astype(jnp.float32)
  amax = jnp.max(jnp.abs(xf), axis=1)  # [M]
  scale = _guard_scale(amax / E4M3_MAX)
  # keepdims-style broadcast at the XLA level (in-kernel [:, None] is the
  # Mosaic trap; here it is plain XLA and fine).
  q = (xf / scale[:, None]).astype(jnp.float8_e4m3fn)
  return q, scale


def quantize_weights(w: jax.Array, block_k: int):
  """Per-K-block amax quantize: w [G, K, N] -> (e4m3 [G, K, N],
  rhs_scale f32 [G, num_blocks, 1, N]) — gmm_v2's rhs_scale contract.

  block_k must be >= MIN_QUANT_BLOCK (the fp8-MXU cliff) and divide K.
  dequant[g, k, n] = q[g, k, n] * rhs_scale[g, k // block_k, 0, n].
  """
  g, k, n = w.shape
  assert k % block_k == 0, (k, block_k)
  assert block_k >= MIN_QUANT_BLOCK, (
      f"quant_block {block_k} < {MIN_QUANT_BLOCK}: the kernel dequantizes "
      "BEFORE the matmul -> silent bf16 matmul (the measured fp8 cliff)"
  )
  nb = k // block_k
  wf = w.astype(jnp.float32).reshape(g, nb, block_k, n)
  amax = jnp.max(jnp.abs(wf), axis=2, keepdims=True)  # [G, NB, 1, N]
  scale = _guard_scale(amax / E4M3_MAX)
  q = (wf / scale).reshape(g, k, n).astype(jnp.float8_e4m3fn)
  return q, scale


def dequantize_weights(w_q: jax.Array, rhs_scale: jax.Array) -> jax.Array:
  """f32 dequant (reference-side helper for check_fp8's XLA-composed math)."""
  g, k, n = w_q.shape
  nb = rhs_scale.shape[1]
  wf = w_q.astype(jnp.float32).reshape(g, nb, k // nb, n)
  return (wf * rhs_scale.astype(jnp.float32)).reshape(g, k, n)
