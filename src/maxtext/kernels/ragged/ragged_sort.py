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
    full_num_slots=None,
    slot_window=None,
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
    full_num_slots: static ``int`` -- the TOTAL flat slot count (num_tokens * topk) of the
      WHOLE combine problem. Only needed when ``topk_argsort_revert_indices`` is a SLICE of
      the full revert permutation: the packed-vs-global buffering-mode decision below is a
      property of the buffer layout (full problem), NOT of the index slice length, so it
      must compare ``buffer_size`` against the FULL slot count. Defaults to
      ``topk_argsort_revert_indices.shape[0]`` (un-chunked call).
    slot_window: optional static ``(start, end)`` pair of Python ints -- restrict this call
      to the contiguous window of output SLOTS ``[start, end)``. ALL operands keep their
      FULL (un-chunked) shapes -- the window only ANDs into the validity mask, so the SC
      kernels see exactly the shapes/contracts of the un-chunked call while doing only the
      window's share of gather/reduce work (their validity compaction skips masked rows).
      Output rows outside ``[start // topk, end // topk)`` are zero. This is how
      decouple_combine_rs_chunks chunks the combine WITHOUT slicing kernel operands.

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
    num_slots = num_tokens if full_num_slots is None else full_num_slots

    # We support two buffering modes:
    # 1. buffer_size >= num_slots: sorted_tokens_local covers ALL flat slots,
    #    and local tokens are at their global positions [start, end).
    # 2. buffer_size < num_slots: sorted_tokens_local is a TRUNCATED buffer,
    #    and local tokens are packed at [0, local_num_tokens).
    # The decision must use the FULL slot count (num_slots), never the possibly-CHUNKED
    # index length: for a chunked call (full_num_slots set), buffer_size can exceed the
    # chunk's slice length while the buffer is still packed/truncated -- comparing against
    # the slice length would mis-read packed local rows at global positions (the
    # decouple_combine_rs_chunks c>0 shard-misalignment bug).
    if slot_window is not None:
      # Static contiguous slot window (decouple_combine_rs_chunks): full-shape operands,
      # validity-restricted work. Slots outside [win_start, win_end) are masked invalid, so
      # the SC kernel's per-partition validity compaction skips them (~window/total work).
      win_start, win_end = slot_window
      slot_pos = jax.lax.iota(jnp.int32, num_tokens)
      window_mask = (slot_pos >= win_start) & (slot_pos < win_end)
    else:
      window_mask = None

    if buffer_size >= num_slots:
      # Express the scatter as a gather-reduce: each output row pulls
      # from sorted_tokens_local at position `topk_argsort_revert_indices[i]` if
      # that position is within this shard's [start, end) range, else zero.
      # The routing weights are applied per-row before the topk reduction.
      valid_rows_mask = (topk_argsort_revert_indices >= shard_output_start) & (
          topk_argsort_revert_indices < shard_output_end
      )
      if window_mask is not None:
        valid_rows_mask = valid_rows_mask & window_mask
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
      if window_mask is not None:
        valid_rows_mask = valid_rows_mask & window_mask
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
    num_slots = n if full_num_slots is None else full_num_slots
    # Build the inverse permutation idx_inv such that idx_inv[j] = i
    # where revert[i] = j.
    idx_inv = jnp.argsort(topk_argsort_revert_indices)

    # Handle the same two buffering modes for backward pass. The mode decision
    # mirrors the forward: it compares against the FULL slot count (num_slots),
    # never the possibly-chunked index slice length n.
    # We let ragged_gather do both the fan-out (by indexing into the
    # un-expanded g_hidden_states_local via idx_inv // topk) and the
    # per-slot weight application (via the fused weights parameter),
    # avoiding an extra HBM read-write pass.
    if buffer_size >= num_slots:
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
      if buffer_size > n:
        # CHUNKED-INPUT combine (decouple_combine_rs_chunks): topk_argsort_revert_indices is a
        # TOKEN-axis SLICE of the full revert permutation, so grad_sorted_tokens has only n rows
        # while the primal input sorted_tokens_local has buffer_size rows. JAX requires the bwd
        # output to match the un-sliced primal shape, so scatter the n grads back into a full
        # buffer_size zero buffer at the buffer positions this slice read. Each chunk reads a
        # DISJOINT set of buffer positions, so the N chunks' grads (summed by autodiff over the
        # shared sorted_tokens_local input) form the complete buffer gradient. No-op when
        # buffer_size == n (un-chunked path is untouched).
        sorted_revert = topk_argsort_revert_indices[idx_inv]
        grad_sorted_tokens = (
            jnp.zeros((buffer_size,) + grad_sorted_tokens.shape[1:], dtype=grad_sorted_tokens.dtype)
            .at[sorted_revert]
            .set(grad_sorted_tokens)
        )
    elif full_num_slots is not None and full_num_slots != n:
      # Truncated (packed) buffer + chunked indices under autodiff: the packed-mode bwd below
      # assumes idx_inv is indexed by buffer position, which only holds for the FULL revert
      # permutation. Supporting this needs a dedicated bwd (see the memory-flat single-custom_vjp
      # design) -- fail loudly instead of silently mis-routing gradients.
      raise NotImplementedError(
          "ring_ragged_unsort backward does not support chunked indices with a truncated "
          "(packed) buffer (decouple_combine_rs_chunks with ragged_buffer_factor > 0 under "
          "autodiff). Use the un-chunked combine, a full-size buffer, or the hand-written "
          "layer backward (moe_handwritten_bwd), whose recompute uses the un-chunked combine."
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

  SLICED-OPERAND FORMULATION (A/B variant vs the full-shape window formulation):
    Each chunk slices ``revert_indices`` / ``topk_weights`` to its contiguous slot range
    (tpu-inference style) and calls ``ring_ragged_unsort`` on the slice; the expert-sorted
    buffer is still read WHOLE (never sliced). Per-chunk SC preprocessing / validity
    compaction is therefore O(T/N) per call -- the window formulation invokes the kernel N
    times at FULL slot-array shapes and pays an O(T) fixed cost per call (measured +1.32s on
    the SC lane at N=4 on v7x, TC unchanged), which is why this variant slices.

    Sliced operands are SAFE only because of three fixes that ride along:
      * ragged_gather_reduce's row-partition stride is derived from the indices operand
        (src_indices_hbm_ref.shape[0], as tpu-inference always did), not from the x buffer --
        pre-fix, x.rows = N x indices.rows made every partition p >= 1 read the slot arrays
        out of bounds (the step-0 NaN at every N > 1).
      * ``full_num_slots`` pins ring_ragged_unsort's packed-vs-global buffering-mode decision
        to the WHOLE problem's slot count (a chunk-length comparison flips modes with a
        truncated buffer).
      * ragged_gather's fallback applies weights in f32 then casts back to x.dtype (the SC
        kernel contract), so the per-chunk bwd cotangent dtype does not fan out.

  EXPERT-SORTED INPUT vs TOKEN-SORTED RS:
    * ``sorted_tokens_local`` (the GMM2 output) is EXPERT-sorted and read WHOLE by every chunk;
      ``ring_ragged_unsort`` emits TOKEN-ordered output, so the expert->token un-sort happens
      INSIDE the combine and the RS only ever sees TOKEN-ordered data.
    * Token order THROUGH the per-chunk RS is corrected by ``_permute_tokens_for_chunked_rs``
      (vLLM trick): reorder the token axis chunk-major BEFORE chunking so that per-chunk
      ``psum_scatter`` + ``concat`` reassembles the correct GLOBAL token order. Without it the
      concat would interleave (shard's slice of chunk 0, then chunk 1, ...) -- wrong order.

  No new custom_vjp: per chunk we reuse the existing ``ring_ragged_unsort`` (its hand-written
  combine bwd, incl. the chunked-input grad scatter-back for buffer_size > n) +
  ``jax.lax.psum_scatter`` (auto all_gather-transpose bwd), so the gradient map is unchanged.

  Returns ``[num_tokens // ep_size, hidden]`` (this shard's reduce-scattered slice).
  """
  num_tokens = topk_argsort_revert_indices.shape[0] // topk
  full_num_slots = topk_argsort_revert_indices.shape[0]

  def _permute_tokens_for_chunked_rs(a):  # a: [num_tokens, topk] -> chunk-major token reorder
    per = num_tokens // (n_chunks * ep_size)
    ar = a.reshape(ep_size, n_chunks, per, a.shape[-1])
    return jnp.transpose(ar, (1, 0, 2, 3)).reshape(a.shape)

  ridx, w = topk_argsort_revert_indices, topk_weights
  if n_chunks > 1 and ep_size > 1:  # reorder the TOKEN axis so per-chunk RS lands in global order
    ridx = _permute_tokens_for_chunked_rs(ridx.reshape(num_tokens, topk)).reshape(-1)
    w = _permute_tokens_for_chunked_rs(w.reshape(num_tokens, topk)).reshape(-1)

  slots_per_chunk = (num_tokens // n_chunks) * topk
  outs = []
  for c in range(n_chunks):
    s0, s1 = c * slots_per_chunk, (c + 1) * slots_per_chunk
    # combine: FULL expert-sorted buffer in (unsliced), sliced indices/weights -> O(T/N)
    # per-chunk kernel preprocessing; token-ordered chunk out (existing custom_vjp)
    combined = ring_ragged_unsort(
        sorted_tokens_local,
        group_sizes_local,
        ridx[s0:s1],
        topk,
        local_num_experts,
        ep_name,
        topk_weights=w[s0:s1],
        full_num_slots=full_num_slots,
        **unsort_kwargs,
    )
    # reduce-scatter the TOKEN-ordered chunk over the expert axis (auto bwd = all_gather)
    outs.append(jax.lax.psum_scatter(combined, ep_name, scatter_dimension=0, tiled=True))
  return jnp.concatenate(outs, axis=0)


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
