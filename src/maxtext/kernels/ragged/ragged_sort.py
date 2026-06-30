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

"""Ragged token sorting operations with custom VJP."""

import jax
import jax.numpy as jnp
from maxtext.kernels.ragged.ragged_gather import ragged_gather
from maxtext.kernels.ragged.ragged_gather_reduce import ragged_gather_reduce


def ring_ragged_sort(
    hidden_states_local,
    topk_indices_local,
    num_experts,
    topk,
    ep_name,
    ep_size,
    buffer_size=None,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
):
  """Ragged-gather variant for AG-RS Expert Parallelism token routing.

  Unlike :func:`a2a_ragged_sort`, which operates on a valid prefix within a single shard,
  this function sorts and gathers tokens across the global expert space but extracts
  only the specific range of output rows assigned to the experts residing on this shard.
  The rest of the shard's output buffer is padded with zeros.

  Forward:
    Sorts tokens based on ``topk_indices_local`` and gathers
    ``hidden_states_local[token_indices_sorted[i]]`` into ``out[i]`` for
    ``i`` in ``[shard_output_start, shard_output_end)``; other rows are zeroed.

  Backward (gather-reduce):
    ``g_hidden_states[token_indices_sorted[i]] += g_out[i]`` for
    ``i`` in ``[shard_output_start, shard_output_end)``. Because each input token
    contributes to exactly ``topk`` experts, this maps cleanly to a
    ``ragged_gather_reduce`` with ``reduce_group_size=topk`` along the inverse
    permutation.

  Args:
    hidden_states_local: 2D ``[num_tokens_local, hidden]`` input tensor.
    topk_indices_local: 2D ``[num_tokens_local, topk]`` tensor of target expert indices.
    num_experts: scalar ``int`` representing the total global number of experts.
    topk: scalar ``int`` representing the routing top-k factor.
    ep_name: ``str`` identifying the expert parallel axis name.
    ep_size: scalar ``int`` representing the expert parallel mesh size.
    buffer_size: optional scalar ``int`` representing the size of the local buffer.

  Returns:
    A tuple containing:
      - Processed activations tensor of shape ``[buffer_size, hidden]``
      containing
        only the tokens destined for local experts, padded with zeros elsewhere.
      - 1D tensor ``group_sizes_local`` tracking expert token counts.
      - 1D tensor ``topk_argsort_revert_indices`` for inverse routing.
  """

  @jax.custom_vjp
  def _ring_ragged_sort(hidden_states_local, topk_indices_local):
    """Sort and gather activations to different EP shards."""
    return _ring_ragged_sort_fwd(hidden_states_local, topk_indices_local)[0]

  @jax.named_scope("ragged-sort-fwd")
  def _ring_ragged_sort_fwd(hidden_states_local, topk_indices_local):
    """Sort and gather activations forward pass."""

    num_tokens_local = hidden_states_local.shape[0]

    topk_indices_flat = topk_indices_local.flatten()  # num_tokens_local x topk
    topk_argsort_indices = jnp.argsort(topk_indices_flat)  # num_tokens_local x topk

    token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)  # num_tokens_local x topk
    token_indices_sorted = token_indices[topk_argsort_indices]  # num_tokens_local x topk

    group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)  # GLOBAL_NUM_EXPERTS

    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)  # num_tokens_local x topk
    shard_idx = jax.lax.axis_index(ep_name)

    local_num_experts = num_experts // ep_size
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    if buffer_size is None or buffer_size >= num_tokens_local * topk:
      local_buffer_size = num_tokens_local * topk
      x = ragged_gather(
          hidden_states_local,
          token_indices_sorted,
          shard_output_start[None],
          shard_output_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )
    else:
      local_buffer_size = buffer_size
      # We only gather up to the available buffer size or the actual number of
      # tokens destined for this shard's experts, whichever is smaller.
      gather_end = jnp.minimum(shard_output_end - shard_output_start, local_buffer_size)
      # Pad the indices to ensure we can safely slice a block of size `local_buffer_size`
      # starting at `shard_output_start` without going out-of-bounds during compilation.
      padded_token_indices_sorted = jnp.pad(token_indices_sorted, (0, local_buffer_size))
      sliced_indices = jax.lax.dynamic_slice_in_dim(
          padded_token_indices_sorted,
          shard_output_start,
          local_buffer_size,
          axis=0,
      )
      x = ragged_gather(
          hidden_states_local,
          sliced_indices,
          jnp.int32(0)[None],
          gather_end[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )

    out = (x, group_sizes_local, topk_argsort_revert_indices)

    res = (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        local_buffer_size,
        hidden_states_local.shape,
    )

    return out, res

  @jax.named_scope("ragged-sort-bwd")
  def _ring_ragged_sort_bwd(res, g_out):
    """Backward pass for the gather: a Pallas SC ragged gather reduce.
    The forward gathers ``hidden_states_local[token_indices_sorted[i]]`` into
    ``x[i]`` for ``i`` in ``[shard_output_start, shard_output_end)``.  The
    gradient w.r.t. ``hidden_states_local`` is therefore::
        g_hidden_states[token_indices_sorted[i]] += g_x[i]    (i in valid range)
    which is exactly a ragged gather reduce.
    """
    (
        topk_argsort_revert_indices,
        shard_output_start,
        shard_output_end,
        local_buffer_size,
        _,
    ) = res
    g_x, _, _ = g_out
    # Restrict to the [start, end) source range via a validity bitmask. The
    # ragged kernel packs valid rows to the front of each row-partition and
    # only iterates over the populated prefix, so we hand it the mask directly
    # rather than materializing a (mostly-zero) dense buffer ourselves.
    n = topk_argsort_revert_indices.shape[0]

    if local_buffer_size >= n:
      valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
          topk_argsort_revert_indices < shard_output_end
      )
      # The forward scatter-add over `token_indices_sorted` is equivalent to a
      # gather-reduce: each input token has exactly `topk` contributions located
      # at sorted positions `topk_argsort_revert_indices[t*topk:(t+1)*topk]`.
      # `topk_weights` is set to ones because this op has no per-row weighting.
      grad_hidden_states = ragged_gather_reduce(
          g_x,
          topk_argsort_revert_indices,
          topk_weights=jnp.ones((n,), dtype=jnp.float32),
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
      )
    else:
      # Buffering: g_x has size `local_buffer_size` (packed).
      # The revert indices are global [0, n), but they must map to the local
      # packed g_x buffer.
      shifted_indices = topk_argsort_revert_indices - shard_output_start
      local_num_tokens = shard_output_end - shard_output_start
      # We only reduce gradients from the valid portion of the local buffer.
      limit = jnp.minimum(local_num_tokens, local_buffer_size)
      # Mask out tokens that were not gathered (either because they belong to
      # other shards, or they exceeded the local buffer size).
      valid_rows_mask = (shifted_indices >= 0) & (shifted_indices < limit)
      # Clamp invalid indices to 0 to prevent compile-time/run-time out-of-bounds
      # in JAX. These clamped values will be ignored due to `valid_rows_mask`.
      safe_indices = jnp.where(valid_rows_mask, shifted_indices, 0)

      grad_hidden_states = ragged_gather_reduce(
          g_x,
          safe_indices,
          topk_weights=jnp.ones((n,), dtype=jnp.float32),
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
      )
    return grad_hidden_states, None

  _ring_ragged_sort.defvjp(_ring_ragged_sort_fwd, _ring_ragged_sort_bwd)

  return _ring_ragged_sort(hidden_states_local, topk_indices_local)


def ring_ragged_unsort(
    sorted_tokens_local,
    group_sizes_local,
    topk_argsort_revert_indices,
    topk,
    local_num_experts,
    ep_name,
    topk_weights,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
):
  """Dual of :func:`ring_ragged_sort`.

  Forward:
    ``out[i] = sum_k( w[i*topk+k] * sorted_tokens_local[topk_argsort_revert_indices[i*topk+k]] )``
    for positions where ``topk_argsort_revert_indices`` is in
    ``[shard_output_start, shard_output_end)``; other rows are zeroed.
    This scatters the processed outputs from the experts hosted on this shard
    back to their flat arrival buffer positions, applying routing weights
    during the reduction.

  Backward:
    ``g_sorted_tokens[j] = w[i] * g_out[i // topk]`` where
    ``j = topk_argsort_revert_indices[i]`` and ``j`` is in
    ``[shard_output_start, shard_output_end)``.

  Args:
    sorted_tokens_local: 2D ``[buffer_size, hidden]`` output tensor from local experts.
    group_sizes_local: 1D tensor tracking the token loads per expert.
    topk_argsort_revert_indices: 1D permutation restoring flat token positions.
    topk: scalar ``int`` representing the routing top-k factor.
    local_num_experts: scalar ``int`` representing the count of experts hosted on this shard.
    ep_name: ``str`` identifying the expert parallel axis name.
    topk_weights: ``[num_tokens_local * topk]`` tensor of per-slot routing weights.

  Returns:
    A 2D ``[num_tokens_local, hidden]`` tensor with expert outputs scattered back
    to their original global sequence locations, with unpopulated positions zeroed out.
  """

  @jax.custom_vjp
  def _ring_ragged_unsort(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk_weights_flat):
    """Unsort and scatter activations."""
    return _ring_ragged_unsort_fwd(
        sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk_weights_flat
    )[0]

  @jax.named_scope("ragged-unsort-fwd")
  def _ring_ragged_unsort_fwd(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk_weights_flat):
    """Executes unsorting sending tokens back."""
    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)

    shard_idx = jax.lax.axis_index(ep_name)
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts

    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    buffer_size = sorted_tokens_local.shape[0]
    num_tokens = topk_argsort_revert_indices.shape[0]

    # We support two buffering modes:
    # 1. buffer_size >= num_tokens: sorted_tokens_local has size >= num_tokens,
    #    and local tokens are at their global positions [start, end).
    # 2. buffer_size < num_tokens: sorted_tokens_local has size < num_tokens,
    #    and local tokens are packed at [0, local_num_tokens).
    if buffer_size >= num_tokens:
      # Express the scatter as a gather-reduce: each output row pulls
      # from sorted_tokens_local at position `topk_argsort_revert_indices[i]` if
      # that position is within this shard's [start, end) range, else zero.
      # The routing weights are applied per-row before the topk reduction.
      valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
          topk_argsort_revert_indices < shard_output_end
      )
      out = ragged_gather_reduce(
          sorted_tokens_local,
          topk_argsort_revert_indices,
          topk_weights=topk_weights_flat,
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
      )
    else:
      # Shift indices so they map to the packed local buffer [0, local_num_tokens).
      shifted_indices = topk_argsort_revert_indices - shard_output_start
      local_num_tokens = shard_output_end - shard_output_start
      limit = jnp.minimum(local_num_tokens, buffer_size)
      valid_rows_mask = (shifted_indices >= 0) & (shifted_indices < limit)
      safe_indices = jnp.where(valid_rows_mask, shifted_indices, 0)

      out = ragged_gather_reduce(
          sorted_tokens_local,
          safe_indices,
          topk_weights=topk_weights_flat,
          valid_rows_mask=valid_rows_mask,
          reduce_group_size=topk,
          enforce_fallback=enforce_gather_reduce_fallback,
          flops_override=gather_reduce_flops_override,
          bytes_accessed_override=gather_reduce_bytes_accessed_override,
      )

    res = (
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
    )

    return out, res

  @jax.named_scope("ragged-unsort-bwd")
  def _ring_ragged_unsort_bwd(res, g_out):
    """Backward pass for the scatter with routing weights.

    The forward computes (per output token t, with topk slots k=0..topk-1):
      out[t] = sum_k  w[t*topk+k] * sorted_tokens[revert[t*topk+k]]
    masked by validity.

    Gradient w.r.t. sorted_tokens:
      g_sorted_tokens[j] = w[i] * g_out[i // topk]
    where j = revert[i] and j in [start, end).
    """
    (
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
    ) = res
    g_hidden_states_local = g_out

    n = topk_argsort_revert_indices.shape[0]
    # Build the inverse permutation idx_inv such that idx_inv[j] = i
    # where revert[i] = j.
    idx_inv = jnp.argsort(topk_argsort_revert_indices)

    # Handle the same two buffering modes for backward pass.
    # We let ragged_gather do both the fan-out (by indexing into the
    # un-expanded g_hidden_states_local via idx_inv // topk) and the
    # per-slot weight application (via the fused weights parameter),
    # avoiding an extra HBM read-write pass.
    if buffer_size >= n:
      # ragged_gather fans out g_hidden_states_local by reading the same row
      # multiple times when idx_inv // topk maps multiple positions to it.
      # Per-slot routing weights are applied inside the kernel.
      weight_for_sorted = topk_weights_flat[idx_inv]
      grad_sorted_tokens = ragged_gather(
          g_hidden_states_local,
          idx_inv // topk,
          shard_output_start[None],
          shard_output_end[None],
          weights=weight_for_sorted,
          has_weights=True,
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )
    else:
      # Slice the inverse permutation to match the packed local buffer.
      padded_idx_inv = jnp.pad(idx_inv, (0, buffer_size))
      sliced_idx_inv = jax.lax.dynamic_slice_in_dim(padded_idx_inv, shard_output_start, buffer_size, axis=0)
      gather_end = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
      # Slice the per-slot routing weights to match the packed local buffer.
      padded_weights = jnp.pad(topk_weights_flat[idx_inv], (0, buffer_size))
      sliced_weights = jax.lax.dynamic_slice_in_dim(padded_weights, shard_output_start, buffer_size, axis=0)
      grad_sorted_tokens = ragged_gather(
          g_hidden_states_local,
          sliced_idx_inv // topk,
          jnp.int32(0)[None],
          gather_end[None],
          weights=sliced_weights,
          has_weights=True,
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )
    return grad_sorted_tokens, None, None, None

  _ring_ragged_unsort.defvjp(_ring_ragged_unsort_fwd, _ring_ragged_unsort_bwd)

  # Build the flat weights array from the routing weights.
  topk_weights_flat = topk_weights.astype(jnp.float32)

  return _ring_ragged_unsort(sorted_tokens_local, group_sizes_local, topk_argsort_revert_indices, topk_weights_flat)


def chunked_ring_combine_reduce_scatter(
    sorted_tokens_local,
    group_sizes_local,
    topk_argsort_revert_indices,
    topk,
    local_num_experts,
    ep_name,
    topk_weights,
    ep_size,
    n_chunks,
    **unsort_kwargs,
):
  """Decoupled chunked combine -> reduce-scatter (ring-of-experts).

  Runs the GMM FULL and chunks ONLY the post-GMM combine+RS loop, so each chunk's reduce-scatter
  hides under the next chunk's combine (validated on v7x: ~11.5ms combine-bound region with the RS
  absorbed). This is DISTINCT from ``moe_n_chunks``, which chunks the whole body (GMM included) and
  so caps at ~2 on GMM efficiency -- here the GMM is untouched, so n_chunks can go large.

  EXPERT-SORTED INPUT vs TOKEN-SORTED RS -- the subtle part, handled explicitly:
    * ``sorted_tokens_local`` (the GMM2 output) is EXPERT-sorted, and is passed WHOLE to every
      chunk -- it is NEVER sliced. ``ring_ragged_unsort`` reads that full expert-sorted buffer via
      ``revert_indices`` and emits TOKEN-ordered output, so the expert->token un-sort happens
      INSIDE the combine on the complete buffer.
    * We chunk only the TOKEN (output) axis, by slicing ``revert_indices`` / ``topk_weights`` (a
      contiguous range of output tokens). Each chunk's combine gathers its tokens' scattered rows
      from the full buffer -> token-ordered chunk -> ``psum_scatter``. The RS therefore only ever
      sees TOKEN-ordered data; there is NO expert/token mismatch at the reduce-scatter.
    * Token order THROUGH the per-chunk RS is corrected by ``_permute_tokens_for_chunked_rs``
      (vLLM trick): reorder the token axis chunk-major BEFORE chunking so that per-chunk
      ``psum_scatter`` + ``concat`` reassembles the correct GLOBAL token order. Without it the
      concat would interleave (shard's slice of chunk 0, then chunk 1, ...) -- wrong order.

  SINGLE custom_vjp for N-independent backward memory:
    The naive approach (per-chunk ``ring_ragged_unsort`` custom_vjp + autodiff) makes the backward
    materialize N separate full ``buffer_size`` grad buffers and SUM them, so backward HBM scales
    with N (N=8 OOMs at mini-DSv3 PBS2). Instead we wrap the WHOLE chunked combine in ONE custom_vjp
    whose backward MANUALLY ACCUMULATES ``grad_sorted_tokens`` into a SINGLE ``buffer_size`` buffer
    (``acc = acc.at[...].add(...)`` per chunk), so backward HBM is ~2 buffers regardless of N. The
    FORWARD is byte-for-byte the same unrolled per-chunk loop (RS-under-next-combine overlap intact);
    only the backward differs from naive autodiff. Per-chunk backward is exactly the mode-1
    ``_ring_ragged_unsort_bwd`` math (all_gather of the chunk's g_out, ragged_gather fan-out with
    per-slot weights, scatter to buffer positions); chunks read DISJOINT buffer positions so the
    running scatter-add over chunks reproduces the full-buffer gradient.

  Validated end-to-end: chunked output == un-chunked (single combine + single RS),
  max_abs 7.8e-3 (bf16 rounding only) at n_chunks 2/4.

  Returns ``[num_tokens // ep_size, hidden]`` (this shard's reduce-scattered slice).
  """
  num_tokens = topk_argsort_revert_indices.shape[0] // topk
  enforce_gather_fallback = unsort_kwargs.get("enforce_gather_fallback", False)

  def _permute_tokens_for_chunked_rs(a):  # a: [num_tokens, topk] -> chunk-major token reorder
    per = num_tokens // (n_chunks * ep_size)
    ar = a.reshape(ep_size, n_chunks, per, a.shape[-1])
    return jnp.transpose(ar, (1, 0, 2, 3)).reshape(a.shape)

  ridx, w = topk_argsort_revert_indices, topk_weights
  if n_chunks > 1 and ep_size > 1:  # reorder the TOKEN axis so per-chunk RS lands in global order
    ridx = _permute_tokens_for_chunked_rs(ridx.reshape(num_tokens, topk)).reshape(-1)
    w = _permute_tokens_for_chunked_rs(w.reshape(num_tokens, topk)).reshape(-1)

  slots_per_chunk = (num_tokens // n_chunks) * topk
  rows_per_chunk_out = (num_tokens // n_chunks) // ep_size  # RS shrinks token axis by ep_size
  # NOTE: the bwd does NOT save sorted_tokens_local as a residual (it needs neither its values nor its
  # shape) -- mirroring ring_ragged_unsort, whose bwd recomputes grad from g_out + indices only. This
  # keeps the ~15GB buffer out of the 8-layer remat-scan residuals.

  # ridx / w are passed as EXPLICIT custom_vjp inputs (not closed-over): JAX forbids differentiating
  # a custom_vjp wrt a closed-over differentiable value, and ``w`` (the router weights) IS
  # differentiable upstream. Like ring_ragged_unsort, we return None for their cotangents (no gradient
  # flows to the routing weights through the combine -- matching the un-chunked path exactly).
  @jax.custom_vjp
  def _chunked_combine_rs(sorted_tokens_local, group_sizes_local, ridx, w):
    return _chunked_combine_rs_fwd(sorted_tokens_local, group_sizes_local, ridx, w)[0]

  @jax.named_scope("chunked-combine-rs-fwd")
  def _chunked_combine_rs_fwd(sorted_tokens_local, group_sizes_local, ridx, w):
    # FORWARD: unrolled per-chunk combine + reduce-scatter. Identical to the naive forward so the
    # RS-under-next-combine overlap (the whole point of the lever) is preserved exactly.
    outs = []
    for c in range(n_chunks):
      s0, s1 = c * slots_per_chunk, (c + 1) * slots_per_chunk
      # combine: FULL expert-sorted buffer in (unsliced), token-ordered chunk out
      combined = ring_ragged_unsort(
          sorted_tokens_local,
          group_sizes_local,
          ridx[s0:s1],
          topk,
          local_num_experts,
          ep_name,
          topk_weights=w[s0:s1],
          **unsort_kwargs,
      )
      # reduce-scatter the TOKEN-ordered chunk over the expert axis
      outs.append(jax.lax.psum_scatter(combined, ep_name, scatter_dimension=0, tiled=True))
    out = jnp.concatenate(outs, axis=0)
    # Residuals: group sizes (offsets) + permuted ridx / w (slot->buffer map + per-slot weights).
    # NOT sorted_tokens_local -- the bwd recomputes grad from g_out + indices alone (mirroring
    # ring_ragged_unsort), so saving the ~15GB buffer here is unnecessary and would balloon the
    # 8-layer remat-scan residual memory.
    return out, (group_sizes_local, ridx, w)

  @jax.named_scope("chunked-combine-rs-bwd")
  def _chunked_combine_rs_bwd(res, g_out):
    # BACKWARD: produce grad_sorted_tokens (one buffer_size buffer) WITHOUT a generic scatter.
    # The N chunks together form the full (permuted) combine, so the combine's bwd is exactly the
    # un-chunked ring_ragged_unsort mode-1 bwd applied to the FULL permuted ridx/w. We therefore:
    #   (1) per chunk: undo psum_scatter (all_gather of its g_out slice) -> token-ordered chunk grad,
    #       concat all chunks -> g_combined_full ([num_tokens, hidden], permuted token order);
    #   (2) ONE ragged_gather over the full buffer (NOT a per-slot scatter): output row j = buffer
    #       position j, masked to this shard's [start,end). This is the proven un-chunked bwd math,
    #       and uses the SparseCore ragged_gather (no scatter_p), so it compiles/schedules like the
    #       un-chunked path. Single buffer_size output -> backward HBM is N-independent.
    group_sizes_local, ridx, w = res

    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_idx = jax.lax.axis_index(ep_name)
    experts_start = shard_idx * local_num_experts
    experts_end = experts_start + local_num_experts
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_end]

    # (1) all_gather each chunk's g_out slice (psum_scatter transpose) and concat in chunk order ->
    # the full token-ordered combine cotangent, matching the permuted ridx/w token order.
    g_chunks = [
        jax.lax.all_gather(
            g_out[c * rows_per_chunk_out : (c + 1) * rows_per_chunk_out], ep_name, axis=0, tiled=True
        )
        for c in range(n_chunks)
    ]
    g_combined_full = jnp.concatenate(g_chunks, axis=0)  # [num_tokens, hidden]

    # (2) un-chunked mode-1 bwd on the full permuted revert: buffer-ordered fan-out gather + weights,
    # masked to this shard's expert output range. idx_inv[j] = slot i with ridx[i] == j (buffer pos j).
    idx_inv = jnp.argsort(ridx)
    weight_for_sorted = w[idx_inv]
    grad_sorted_tokens = ragged_gather(
        g_combined_full,
        idx_inv // topk,
        shard_output_start[None],
        shard_output_end[None],
        weights=weight_for_sorted,
        has_weights=True,
        enforce_fallback=enforce_gather_fallback,
    )
    # Cotangents: grad for sorted_tokens_local; None for group_sizes_local / ridx / w (no grad flows
    # to the routing weights through the combine, matching the un-chunked ring_ragged_unsort path).
    return grad_sorted_tokens, None, None, None

  _chunked_combine_rs.defvjp(_chunked_combine_rs_fwd, _chunked_combine_rs_bwd)

  return _chunked_combine_rs(sorted_tokens_local, group_sizes_local, ridx, w)


def a2a_ragged_sort(inputs, sort_indices, valid_end, enforce_gather_fallback=False, enforce_gather_reduce_fallback=False):
  """Ragged-gather variant for ``local_permute``.

  Unlike :func:`ring_ragged_sort`, the rows valid for this shard live in
  the prefix ``[0, valid_end)`` of ``inputs`` (the rest of the buffer is
  padding). This helper sorts ``inputs`` by ``sort_indices`` but only touches
  the valid prefix, making the cost proportional to the *actual* token count
  rather than the padded buffer length.

  Forward:
    ``out[i] = inputs[sort_indices[i]]`` for ``i in [0, valid_end)``;
    other rows are zero.

  Backward (gather-reduce):
    ``g_inputs[sort_indices[i]] += g_out[i]`` for ``i in [0, valid_end)``.
    Because ``sort_indices`` is a permutation, each input row receives exactly
    one contribution; we model this as a ``ragged_gather_reduce`` with
    ``reduce_group_size=1`` along the inverse permutation.

  Args:
    inputs: 2D ``[num_tokens, hidden]`` tensor whose valid rows live in the
      prefix ``[0, valid_end)``.
    sort_indices: 1D permutation of ``[0, num_tokens)`` describing the desired
      ordering. Values at positions ``>= valid_end`` are ignored.
    valid_end: scalar ``int32`` indicating the exclusive end of the valid
      prefix.

  Returns:
    A 2D ``[num_tokens, hidden]`` tensor sorted by ``sort_indices`` over the
    valid prefix, with padded rows zeroed.
  """

  @jax.custom_vjp
  def _a2a_ragged_sort(inputs, sort_indices, valid_end):
    return _a2a_ragged_sort_fwd(inputs, sort_indices, valid_end)[0]

  @jax.named_scope("local-ragged-sort-fwd")
  def _a2a_ragged_sort_fwd(inputs, sort_indices, valid_end):
    start = jnp.int32(0)
    end = valid_end.astype(jnp.int32) if hasattr(valid_end, "astype") else jnp.int32(valid_end)
    out = ragged_gather(inputs, sort_indices, start[None], end[None])
    n = sort_indices.shape[0]
    valid_mask = jnp.arange(n) < end
    out = jnp.where(valid_mask[:, None], out, 0.0)
    res = (sort_indices, end, inputs.shape)
    return out, res

  @jax.named_scope("local-ragged-sort-bwd")
  def _a2a_ragged_sort_bwd(res, g_out):
    sort_indices, end, _ = res
    n = sort_indices.shape[0]
    valid_rows_mask = jnp.arange(n) < end
    # g_inputs[sort_indices[i]] += g_out[i], for i in [0, end). This is a
    # ragged scatter-add, which we express as a gather-reduce along the inverse
    # permutation: each input row j receives exactly one contribution from
    # output row i where sort_indices[i] == j.
    idx_inv = jnp.argsort(sort_indices)
    grad_inputs = ragged_gather_reduce(
        g_out,
        idx_inv,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask[idx_inv],
        reduce_group_size=1,
        enforce_fallback=enforce_gather_reduce_fallback,
    )
    # custom_vjp must return one gradient per primal arg; valid_end is integer
    # and non-differentiable, so we return None for it.
    return grad_inputs, None, None

  _a2a_ragged_sort.defvjp(_a2a_ragged_sort_fwd, _a2a_ragged_sort_bwd)
  return _a2a_ragged_sort(inputs, sort_indices, valid_end)


def a2a_ragged_unsort(
    sorted_tokens, revert_indices, valid_end, enforce_gather_fallback=False, enforce_gather_reduce_fallback=False
):
  """Dual of :func:`a2a_ragged_sort`.

  Forward:
    ``out[i] = sorted_tokens[revert_indices[i]]`` for ``i in [0, valid_end)``;
    other rows are zero. This is the unsort step in the local-permute path:
    given a buffer ordered by local expert IDs, restore the original arrival
    order.

  Backward:
    ``g_sorted_tokens[j] = g_out[i]`` where ``j = revert_indices[i]``, for
    ``i in [0, valid_end)``. Since ``revert_indices`` is a permutation, this
    is a simple ragged gather over the inverse permutation.

  Args:
    sorted_tokens: 2D ``[num_tokens, hidden]`` tensor.
    revert_indices: 1D permutation of ``[0, num_tokens)``.
    valid_end: scalar ``int32`` indicating the exclusive end of the valid
      prefix.

  Returns:
    A 2D ``[num_tokens, hidden]`` tensor with rows reordered by
    ``revert_indices`` over the valid prefix and zero elsewhere.
  """

  @jax.custom_vjp
  def _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end):
    return _a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end)[0]

  @jax.named_scope("local-ragged-unsort-fwd")
  def _a2a_ragged_unsort_fwd(sorted_tokens, revert_indices, valid_end):
    start = jnp.int32(0)
    end = valid_end.astype(jnp.int32) if hasattr(valid_end, "astype") else jnp.int32(valid_end)
    n = revert_indices.shape[0]
    valid_rows_mask = jnp.arange(n) < end
    out = ragged_gather_reduce(
        sorted_tokens,
        revert_indices,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask,
        reduce_group_size=1,
        enforce_fallback=enforce_gather_reduce_fallback,
    )
    res = (revert_indices, end, sorted_tokens.shape, start)
    return out, res

  @jax.named_scope("local-ragged-unsort-bwd")
  def _a2a_ragged_unsort_bwd(res, g_out):
    revert_indices, end, sorted_tokens_shape, start = res
    # g_sorted_tokens[revert_indices[i]] = g_out[i] for i in [0, end).
    # Because revert_indices is a permutation, build the inverse and use
    # ragged_gather to pull the per-row gradients to the right positions.
    idx_inv = jnp.argsort(revert_indices)
    grad_sorted = ragged_gather(g_out, idx_inv, start[None], end[None])
    num_rows = sorted_tokens_shape[0]
    pos = jnp.arange(num_rows)
    valid = pos < end
    grad_sorted = jnp.where(valid[:, None], grad_sorted, jnp.zeros_like(grad_sorted))

    return grad_sorted, None, None

  _a2a_ragged_unsort.defvjp(_a2a_ragged_unsort_fwd, _a2a_ragged_unsort_bwd)
  return _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end)
