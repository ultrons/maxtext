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
from maxtext.kernels.ragged.ragged_gather_reduce_v2 import ragged_gather_reduce


def compute_ring_sort_indices(topk_indices_local, num_experts, topk):
  """The integer index bundle of ``ring_ragged_sort``'s forward, computed standalone.

  Returns ``(token_indices_sorted, group_sizes_local, topk_argsort_revert_indices)`` --
  exactly the tensors ``ring_ragged_sort`` derives internally from ``topk_indices_local``
  (two argsorts + a one-hot group-size sum; all int32, data-independent of the hidden
  states). Used by moe_save_sort_indices: the forward computes the bundle ONCE here,
  feeds it to ``ring_ragged_sort(precomputed_sort=...)`` (bit-identical output), and
  saves it so the hand-written backward's recompute skips these index computations.
  """
  num_tokens_local = topk_indices_local.shape[0]
  topk_indices_flat = topk_indices_local.flatten()  # num_tokens_local x topk
  topk_argsort_indices = jnp.argsort(topk_indices_flat)  # num_tokens_local x topk
  token_indices = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)  # num_tokens_local x topk
  token_indices_sorted = token_indices[topk_argsort_indices]  # num_tokens_local x topk
  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)  # GLOBAL_NUM_EXPERTS
  topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)  # num_tokens_local x topk
  return token_indices_sorted, group_sizes_local, topk_argsort_revert_indices


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
    use_single_sparsecore=False,
    precomputed_sort=None,
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

  def _sorted_gather(hidden_states_local, token_indices_sorted, group_sizes_local):
    """The data gather of the ring ragged sort, given the (int) sort bundle.

    Returns ``(x, shard_output_start, shard_output_end, local_buffer_size)``; shared by the
    stock forward (bundle computed in-line) and the precomputed_sort forward (bundle saved
    from an earlier trace)."""
    num_tokens_local = hidden_states_local.shape[0]
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
          use_single_sparsecore=use_single_sparsecore,
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
          use_single_sparsecore=use_single_sparsecore,
      )
    return x, shard_output_start, shard_output_end, local_buffer_size

  @jax.custom_vjp
  def _ring_ragged_sort(hidden_states_local, topk_indices_local):
    """Sort and gather activations to different EP shards."""
    return _ring_ragged_sort_fwd(hidden_states_local, topk_indices_local)[0]

  @jax.named_scope("ragged-sort-fwd")
  def _ring_ragged_sort_fwd(hidden_states_local, topk_indices_local):
    """Sort and gather activations forward pass."""
    token_indices_sorted, group_sizes_local, topk_argsort_revert_indices = compute_ring_sort_indices(
        topk_indices_local, num_experts, topk
    )
    x, shard_output_start, shard_output_end, local_buffer_size = _sorted_gather(
        hidden_states_local, token_indices_sorted, group_sizes_local
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
    # fp8 wire (moe_fp8_dispatch_wire): the forward may dispatch an e4m3 tensor, so the cotangent
    # is 8-bit-float-typed. Gradients flow in bf16 regardless of the forward wire dtype -- normalize
    # so the gather-reduce (which multiplies by f32 weights) doesn't hit an 8-bit promotion. No-op
    # for the normal bf16 path; leaves f32 configs untouched.
    if jnp.issubdtype(g_x.dtype, jnp.floating) and g_x.dtype.itemsize < 2:
      g_x = g_x.astype(jnp.bfloat16)
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
          use_single_sparsecore=use_single_sparsecore,
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
          use_single_sparsecore=use_single_sparsecore,
      )
    return grad_hidden_states, None

  _ring_ragged_sort.defvjp(_ring_ragged_sort_fwd, _ring_ragged_sort_bwd)

  # precomputed_sort (moe_save_sort_indices): the int index bundle is an EXPLICIT custom_vjp
  # input, never a closure -- in the hand-written backward it arrives as a residual-derived
  # (scan-carried) tracer, and a custom_vjp closing over such a tracer hits the
  # "No constant handler for DynamicJaxprTracer" constvar wall. Integer inputs get None
  # cotangents; the hidden-states gradient is byte-for-byte the stock bwd (same residuals).
  @jax.custom_vjp
  def _ring_ragged_sort_pre(hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices):
    return _ring_ragged_sort_pre_fwd(
        hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices
    )[0]

  @jax.named_scope("ragged-sort-pre-fwd")
  def _ring_ragged_sort_pre_fwd(
      hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices
  ):
    x, shard_output_start, shard_output_end, local_buffer_size = _sorted_gather(
        hidden_states_local, token_indices_sorted, group_sizes_local
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

  def _ring_ragged_sort_pre_bwd(res, g_out):
    grad_hidden_states, _ = _ring_ragged_sort_bwd(res, g_out)
    return grad_hidden_states, None, None, None

  _ring_ragged_sort_pre.defvjp(_ring_ragged_sort_pre_fwd, _ring_ragged_sort_pre_bwd)

  if precomputed_sort is not None:
    token_indices_sorted, group_sizes_local, topk_argsort_revert_indices = precomputed_sort
    return _ring_ragged_sort_pre(
        hidden_states_local, token_indices_sorted, group_sizes_local, topk_argsort_revert_indices
    )
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
    use_single_sparsecore=False,
    full_num_slots=None,
    slot_window=None,
    bwd_col_scale=None,
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
  def _ring_ragged_unsort(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      col_scale,
  ):
    """Unsort and scatter activations."""
    return _ring_ragged_unsort_fwd(
        sorted_tokens_local,
        group_sizes_local,
        topk_argsort_revert_indices,
        topk_weights_flat,
        col_scale,
    )[0]

  @jax.named_scope("ragged-unsort-fwd")
  def _ring_ragged_unsort_fwd(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      col_scale,
  ):
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
          use_single_sparsecore=use_single_sparsecore,
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
          use_single_sparsecore=use_single_sparsecore,
      )

    res = (
        topk_argsort_revert_indices,
        topk_weights_flat,
        shard_output_start,
        shard_output_end,
        buffer_size,
        col_scale,
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
        col_scale,
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
      gather_start, gather_end = shard_output_start, shard_output_end
      if buffer_size > n:
        # CHUNKED-INPUT bwd (decouple_combine_rs_chunks / moe_chunked_combine_in_remat):
        # ragged_gather's [start, end) bound ranges over its OUTPUT/INDEX-ROW axis (its kernel
        # derives the processed block range as start // block_size .. cdiv(end, block_size) and
        # indexes indices/weights/out rows with it). Un-chunked, output row j IS buffer position
        # j (n == buffer_size), so passing the shard's BUFFER-POSITION range is correct. For a
        # chunk (n < buffer_size) the output rows are the RANKS of the chunk's slots in
        # buffer-position order (idx_inv sorts them), so the buffer-position range must be
        # converted to RANK space: passing raw buffer positions makes the kernel's block range
        # index the chunk-sized arrays OUT OF BOUNDS (silent -- bounds checks off) or select
        # the wrong rank rows. This was the step-1 NaN under moe_chunked_combine_in_remat: the
        # exact backward mirror of the forward row-partition-stride bug. sorted_revert is
        # ascending, so ranks with position in [start, end) form the contiguous range
        # [searchsorted(start), searchsorted(end)).
        sorted_revert_positions = topk_argsort_revert_indices[idx_inv]
        gather_start = jnp.searchsorted(sorted_revert_positions, shard_output_start).astype(jnp.int32)
        gather_end = jnp.searchsorted(sorted_revert_positions, shard_output_end).astype(jnp.int32)
      grad_sorted_tokens = ragged_gather(
          g_hidden_states_local,
          idx_inv // topk,
          gather_start[None],
          gather_end[None],
          weights=weight_for_sorted,
          has_weights=True,
          # bwd_col_scale: fold the consumer's per-output-channel scale into THIS kernel. It rides
          # the unpack/multiply/repack pass the per-row weights already run, so it costs no extra
          # HBM traffic -- and it removes a whole full-buffer elementwise pass downstream
          # (select_multiply_fusion.3). The consumer must then NOT apply the scale again.
          col_scale=col_scale,
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
          use_single_sparsecore=use_single_sparsecore,
      )
      # sanitize-v3 (the cv-wag real-data NaN door): the SC ragged_gather writes ONLY rows in
      # [gather_start, gather_end) -- its validity compaction SKIPS the rest, leaving stale HBM in
      # this cotangent buffer (the delta operand of the weight-grad tgmm, which contracts over ALL
      # rows: stale Inf/NaN x finite = NaN weight grads; allocation-layout-dependent, hence the
      # print-perturbable heisenbug). The PACKED mode below has always masked its tail -- this
      # mirrors that mask for the full-buffer mode. Must run BEFORE the chunked scatter-back,
      # which would otherwise scatter the stale rows into the full buffer.
      #
      # MOE_UNSORT_BWD_MASK=0 drops the mask; MOE_UNSORT_BWD_POISON=1 fills the unwritten rows
      # with NaN instead of 0. The poison mode is how "is the mask still needed?" becomes a proof
      # rather than a lottery: those rows hold whatever HBM happened to contain, so a run that is
      # merely clean with the mask removed only says this allocation was lucky. Poisoning them
      # deterministically means a finite run proves no consumer reads them. Always pair with a
      # dense-quant positive control, which must NaN -- otherwise the probe proves nothing because
      # the poison never reached a consumer in the first place.
      import os as _os

      _mask_on = _os.environ.get("MOE_UNSORT_BWD_MASK", "1") == "1"
      _poison = _os.environ.get("MOE_UNSORT_BWD_POISON", "0") == "1"
      if _mask_on or _poison:
        _row = jnp.arange(grad_sorted_tokens.shape[0], dtype=jnp.int32)
        _valid = (_row >= gather_start) & (_row < gather_end)
        _fill = jnp.asarray(jnp.nan if _poison else 0.0, grad_sorted_tokens.dtype)
        grad_sorted_tokens = jnp.where(_valid[:, None], grad_sorted_tokens, _fill)
      if buffer_size > n:
        # CHUNKED-INPUT combine (decouple_combine_rs_chunks): topk_argsort_revert_indices is a
        # TOKEN-axis SLICE of the full revert permutation, so grad_sorted_tokens has only n rows
        # while the primal input sorted_tokens_local has buffer_size rows. JAX requires the bwd
        # output to match the un-sliced primal shape, so scatter the n grads back into a full
        # buffer_size zero buffer at the buffer positions this slice read. Each chunk reads a
        # DISJOINT set of buffer positions, so the N chunks' grads (summed by autodiff over the
        # shared sorted_tokens_local input) form the complete buffer gradient. No-op when
        # buffer_size == n (un-chunked path is untouched).
        grad_sorted_tokens = (
            jnp.zeros((buffer_size,) + grad_sorted_tokens.shape[1:], dtype=grad_sorted_tokens.dtype)
            .at[sorted_revert_positions]
            .set(grad_sorted_tokens)
        )
    elif full_num_slots is not None and full_num_slots != n:
      # Truncated (packed) buffer + chunked indices under autodiff: the packed-mode bwd below
      # assumes idx_inv is indexed by buffer position, which only holds for the FULL revert
      # permutation. The chunked combine no longer reaches this branch (its OUTER memory-flat
      # custom_vjp in chunked_ring_combine_reduce_scatter owns the gradient and handles the
      # packed mode with the full permutation) -- fail loudly for any other direct caller
      # instead of silently mis-routing gradients.
      raise NotImplementedError(
          "ring_ragged_unsort backward does not support chunked indices with a truncated "
          "(packed) buffer. Use the un-chunked combine, a full-size buffer, or "
          "chunked_ring_combine_reduce_scatter (whose single memory-flat custom_vjp backward "
          "handles the packed mode over the full permutation)."
      )
    else:
      if col_scale is not None:
        # The packed/truncated branch never applied col_scale, while the wo gmm still skipped its
        # own _dlhs_scale_grad_by_rhs_scale -- the cotangent would be short a factor of wo_scale.
        # Fail loudly rather than train on a silently mis-scaled gradient.
        raise NotImplementedError(
            "moe_fold_wo_scale_in_gather (bwd_col_scale) is only implemented for the FULL-buffer "
            "unsort backward (ragged_buffer_factor<=0). The packed/truncated branch does not apply "
            "col_scale, and the consumer gmm skips its own scale multiply, so the wo dlhs cotangent "
            "would be missing a factor of wo_scale."
        )
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
          use_single_sparsecore=use_single_sparsecore,
      )
      # Mask out gradients for elements beyond the valid limit of the local buffer.
      limit = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
      mask = jnp.arange(buffer_size) < limit
      grad_sorted_tokens = jnp.where(mask[:, None], grad_sorted_tokens, 0.0)
    return grad_sorted_tokens, None, None, None, None

  _ring_ragged_unsort.defvjp(_ring_ragged_unsort_fwd, _ring_ragged_unsort_bwd)

  # Build the flat weights array from the routing weights.
  topk_weights_flat = topk_weights.astype(jnp.float32)

  return _ring_ragged_unsort(
      sorted_tokens_local,
      group_sizes_local,
      topk_argsort_revert_indices,
      topk_weights_flat,
      bwd_col_scale,
  )


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
    return_first_combine_token=False,
    reduce_scatter_fn=None,
    all_gather_fn=None,
    **unsort_kwargs,
):
  """Decoupled chunked combine -> reduce-scatter (ring-of-experts).

  Runs the GMM FULL and chunks ONLY the post-GMM combine+RS loop, so each chunk's reduce-scatter
  hides under the next chunk's combine (validated on v7x: ~11.5ms combine-bound region with the RS
  absorbed). This is DISTINCT from ``num_moe_token_chunks``, which chunks the whole body (GMM included) and
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

  SINGLE custom_vjp for N-independent backward memory (rung 8, memory-flat backward):
    The naive per-chunk backward (each chunk's ``ring_ragged_unsort`` custom_vjp under
    autodiff -- rung 7) runs N full-buffer SC ragged_gathers and materializes + SUMS N
    separate ``buffer_size`` grad buffers (measured 22.5 s/step vs 15.6 baseline). But the
    N chunks together ARE one big permuted combine, so the WHOLE chunked loop is wrapped
    in ONE custom_vjp whose backward is exactly the un-chunked ``ring_ragged_unsort`` bwd
    applied to the FULL permuted ``ridx``/``w``:
      (1) ONE tiled all_gather of the whole contiguous ``g_out`` (the transpose of BOTH
          psum_scatter and the direct-RS Pallas kernel, see ``_drs_bwd``), with the
          (chunk, shard) -> (shard, chunk) row reorder COMPOSED INTO the bwd gather's
          token indices (``_bwd_ag_row``, pure int arithmetic -- no data transpose, no
          extra cotangent copy). Rung 8b: replaces N smaller per-chunk all_gathers
          (profiled +1866 AG ops / +710ms exposed AG lane) -- the gather reads rows
          bit-identical to the per-chunk concat at single-large-AG cost;
      (2) ONE full-buffer ``ragged_gather`` fan-out with ``idx_inv = argsort(ridx)`` +
          per-slot weights, masked to the shard's [shard_output_start, shard_output_end)
          expert range.
    This is the un-chunked-STYLE call: the index array is the full permutation
    (n == num_slots), so output row j IS buffer position j and the shard's
    buffer-position range is the gather's OUTPUT-ROW [start, end) AS-IS -- no rank-space
    searchsorted conversion and no per-chunk scatter-back (those are only needed for
    chunk-sized index arrays; see ``_ring_ragged_unsort_bwd``). One ``buffer_size``
    output -> backward HBM and SC gather work are N-independent. The FORWARD below is
    byte-for-byte the cluster-proven per-chunk loop (sliced operands, barrier chain,
    RS-under-next-combine overlap intact). ``sorted_tokens_local`` is NOT saved as a
    residual (the bwd needs only g_out + indices), keeping the ~15GB buffer out of the
    remat-scan residuals. No gradient flows to the routing weights through the combine
    (None cotangents), matching the un-chunked path.

  Returns ``[num_tokens // ep_size, hidden]`` (this shard's reduce-scattered slice). With
  ``return_first_combine_token=True``, additionally returns a tiny ``[1, 1]`` SCHEDULING TOKEN
  sliced from the FIRST chunk's pre-RS combined output (moe_shared_after_combine): a consumer
  fenced on the token cannot be scheduled before chunk 0's combine has produced its output,
  but does NOT depend on any reduce-scatter -- used to push the shared-expert GMM into the
  chunk-RS window instead of ahead of the combine phase.
  """
  num_tokens = topk_argsort_revert_indices.shape[0] // topk
  full_num_slots = topk_argsort_revert_indices.shape[0]
  buffer_size = sorted_tokens_local.shape[0]
  # bwd gather kwargs: exactly what ring_ragged_unsort's own bwd would read from unsort_kwargs
  # (.get, not .pop -- the per-chunk forward calls still forward the full unsort_kwargs).
  enforce_gather_fallback = unsort_kwargs.get("enforce_gather_fallback", False)
  gather_flops_override = unsort_kwargs.get("gather_flops_override", -1)
  gather_bytes_accessed_override = unsort_kwargs.get("gather_bytes_accessed_override", -1)

  def _permute_tokens_for_chunked_rs(a):  # a: [num_tokens, topk] -> chunk-major token reorder
    per = num_tokens // (n_chunks * ep_size)
    ar = a.reshape(ep_size, n_chunks, per, a.shape[-1])
    return jnp.transpose(ar, (1, 0, 2, 3)).reshape(a.shape)

  ridx, w = topk_argsort_revert_indices, topk_weights
  if n_chunks > 1 and ep_size > 1:  # reorder the TOKEN axis so per-chunk RS lands in global order
    ridx = _permute_tokens_for_chunked_rs(ridx.reshape(num_tokens, topk)).reshape(-1)
    w = _permute_tokens_for_chunked_rs(w.reshape(num_tokens, topk)).reshape(-1)

  slots_per_chunk = (num_tokens // n_chunks) * topk
  rows_per_chunk_out = (num_tokens // n_chunks) // ep_size  # the RS shrinks the token axis by ep_size

  # ridx / w are EXPLICIT custom_vjp inputs (not closed over): JAX forbids differentiating a
  # custom_vjp wrt a closed-over differentiable value (``w``, the router weights, IS differentiable
  # upstream), and a custom_vjp closing over scan-carried TRACERS hits "No constant handler for
  # DynamicJaxprTracer" inside the per-layer scan (mirrors chunked_ring_dispatch).
  @jax.custom_vjp
  def _chunked_combine_rs(sorted_tokens_local, group_sizes_local, ridx, w):
    return _chunked_combine_rs_fwd(sorted_tokens_local, group_sizes_local, ridx, w)[0]

  @jax.named_scope("chunked-combine-rs-fwd")
  def _chunked_combine_rs_fwd(sorted_tokens_local, group_sizes_local, ridx, w):
    # FORWARD: byte-for-byte the cluster-proven per-chunk loop (rung 6-sliced) -- the custom_vjp
    # wrapper changes only the gradient path.
    outs = []
    prev_combined = None
    first_combine_token = None
    for c in range(n_chunks):
      s0, s1 = c * slots_per_chunk, (c + 1) * slots_per_chunk
      ridx_c, w_c = ridx[s0:s1], w[s0:s1]
      stl_c = sorted_tokens_local
      if prev_combined is not None:
        # UN-FUSE (profile forensics, v7x N=4): XLA horizontally fused the four per-chunk
        # f32->bf16 convert+selects (inside ring_ragged_unsort's wrapper) into ONE multi-output
        # fusion consuming all chunks' kernel outputs, so every chunk's reduce-scatter
        # transitively depended on the LAST combine and all N RS starts batched after it --
        # erasing the RS-under-next-combine overlap. Fence this chunk's INPUTS on the previous
        # chunk's PRE-RS combined output: chunk c's convert then depends on chunk c-1's convert
        # OUTPUT, so no single fusion can contain both, and the combines stay emission-ordered.
        # Deliberately NOT fenced on any RS output (that would serialize the pipeline).
        stl_c, ridx_c, w_c, _ = jax.lax.optimization_barrier((stl_c, ridx_c, w_c, prev_combined))
      # combine: FULL expert-sorted buffer in (unsliced), sliced indices/weights -> O(T/N)
      # per-chunk kernel preprocessing; token-ordered chunk out (existing custom_vjp -- its bwd
      # is unreachable here: the outer custom_vjp owns the gradient).
      # `combined` is already x.dtype: the f32->bf16 convert+select lives INSIDE
      # ring_ragged_unsort (ragged_gather_reduce's wrapper), i.e. UPSTREAM of the barriers here.
      combined = ring_ragged_unsort(
          stl_c,
          group_sizes_local,
          ridx_c,
          topk,
          local_num_experts,
          ep_name,
          topk_weights=w_c,
          full_num_slots=full_num_slots,
          **unsort_kwargs,
      )
      # Barrier the (already-converted) per-chunk output so nothing downstream is fused across
      # chunks, and hand it DIRECTLY to this chunk's psum_scatter (structural separation: the
      # convert's only consumer is chunk c's RS). CAVEAT: the production flag set carries
      # xla_tpu_aggressive_opt_barrier_removal=true, which has stripped optimization_barriers
      # before -- if the fusion re-appears in profiles, the chain above is the first suspect
      # and flipping that flag off for this config is the fallback lever.
      combined = jax.lax.optimization_barrier(combined)
      prev_combined = combined
      if c == 0 and return_first_combine_token:
        # [1, 1] scheduling token: depends (through the barrier / the combine's data) on chunk
        # 0's PRE-RS combined output only -- deliberately NOT on any psum_scatter.
        first_combine_token = jax.lax.slice(combined, (0, 0), (1, 1))
      # reduce-scatter the TOKEN-ordered chunk over the expert axis.
      # reduce_scatter_fn (moe_direct_rs): drop-in psum_scatter replacement -- a (x, chunk_idx)
      # callable (chunk_idx selects a distinct collective_id per concurrently-in-flight chunk RS).
      if reduce_scatter_fn is None:
        outs.append(jax.lax.psum_scatter(combined, ep_name, scatter_dimension=0, tiled=True))
      else:
        outs.append(reduce_scatter_fn(combined, c))
    out = jnp.concatenate(outs, axis=0)
    result = (out, first_combine_token) if return_first_combine_token else out
    # Residuals: group sizes (offsets) + the PERMUTED ridx / w (slot->buffer map + per-slot
    # weights). NOT sorted_tokens_local -- the bwd recomputes grad from g_out + indices alone
    # (mirroring ring_ragged_unsort), keeping the ~15GB buffer out of the remat-scan residuals.
    return result, (group_sizes_local, ridx, w)

  @jax.named_scope("chunked-combine-rs-bwd")
  def _chunked_combine_rs_bwd(res, g):
    # BACKWARD (memory-flat, rung 8): ONE buffer_size grad output regardless of N.
    group_sizes_local, ridx, w = res
    if return_first_combine_token:
      # The [1, 1] scheduling token is consumed only through an optimization_barrier whose
      # token output is discarded (the moe_shared_after_combine deadline fence in moe.py), so
      # its cotangent is identically zero -- dropping it is exact (== stop_gradient on the token).
      g_out, _g_token = g
    else:
      g_out = g

    group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
    shard_idx = jax.lax.axis_index(ep_name)
    experts_start = shard_idx * local_num_experts
    shard_output_start = group_offsets[experts_start]
    shard_output_end = group_offsets[experts_start + local_num_experts]

    # (1) ONE tiled all_gather of the WHOLE contiguous g_out -- the transpose of BOTH
    # jax.lax.psum_scatter and the direct-RS Pallas kernel (same reduce-scatter math; see
    # _drs_bwd in moe.py) -- with the row reorder folded into the gather INDICES (rung 8b).
    # The forward's out is concat_c(psum_scatter(chunk_c)), so shard r's local g_out is
    # CHUNK-major (local row c*rpc+i = shard r's slice of chunk c) and the tiled AG lands
    # SHARD-major: permuted-order token t = c*(ep*rpc) + r*rpc + i sits at AG row
    # r*(N*rpc) + c*rpc + i. The per-chunk-AG construction this replaces (N smaller AGs
    # concatenated in chunk order -- profiled as +1866 AG ops / +710ms exposed AG lane time
    # per step) materialized the permuted-order cotangent directly; rather than transposing
    # the DATA back (an extra num_tokens x hidden cotangent copy -- measured +1 chunk-buffer
    # of temp on the isolated-block AOT), ``_bwd_ag_row`` composes the (chunk, shard) ->
    # (shard, chunk) row map into the bwd gather's token indices: pure int arithmetic on the
    # index array, zero extra HBM buffers. The gather reads the IDENTICAL rows it read from
    # the per-chunk concat -- bit-identical grads at single-large-AG cost.
    #
    # all_gather_fn (moe_direct_combine_ag): drop-in replacement for the tiled EP all_gather -- a
    # single-arg callable that runs the direct-to-owner TC Pallas all-gather (_direct_all_gather in
    # moe.py) instead of the XLA collective, so this big exposed combine-cotangent AG (== .626)
    # rides the TensorCore ICI DMAs rather than the SparseCore all-gather-offload queue and XLA can
    # overlap it with the SC combine work. Numerically == lax.all_gather (bf16). None (flag-off) =>
    # the plain collective, byte-identical.
    if all_gather_fn is None:
      g_full = jax.lax.all_gather(g_out, ep_name, axis=0, tiled=True)  # [num_tokens, hidden]
    else:
      g_full = all_gather_fn(g_out)  # [num_tokens, hidden]

    def _bwd_ag_row(tok):  # permuted-order token t -> its row in the SHARD-major AG output
      if n_chunks == 1 or ep_size == 1:
        return tok  # the two orders coincide (same gating as the forward token permute)
      chunk_rows = ep_size * rows_per_chunk_out  # tokens per chunk
      c, rem = tok // chunk_rows, tok % chunk_rows
      r, i = rem // rows_per_chunk_out, rem % rows_per_chunk_out
      return r * (n_chunks * rows_per_chunk_out) + c * rows_per_chunk_out + i

    # (2) the un-chunked ring_ragged_unsort bwd applied to the FULL permuted revert. idx_inv is
    # the full inverse permutation (n == full_num_slots), so the gather's output row j IS buffer
    # position j and the shard's buffer-position range is the OUTPUT-ROW [start, end) AS-IS --
    # the rank-space searchsorted conversion and per-chunk scatter-back in
    # _ring_ragged_unsort_bwd are only needed for CHUNK-SIZED index arrays and never fire here.
    idx_inv = jnp.argsort(ridx)
    weight_for_sorted = w.astype(jnp.float32)[idx_inv]
    if buffer_size >= full_num_slots:
      grad_sorted_tokens = ragged_gather(
          g_full,
          _bwd_ag_row(idx_inv // topk),
          shard_output_start[None],
          shard_output_end[None],
          weights=weight_for_sorted,
          has_weights=True,
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )
      if buffer_size > full_num_slots:
        # Oversized buffer: rows >= full_num_slots are never referenced by ridx (a permutation
        # of [0, full_num_slots)), so their grad is zero -- pad to the primal buffer shape.
        grad_sorted_tokens = jnp.pad(grad_sorted_tokens, ((0, buffer_size - full_num_slots), (0, 0)))
    else:
      # TRUNCATED (packed, mode-2) buffer: identical math to _ring_ragged_unsort_bwd's packed
      # branch. Valid here -- unlike the per-chunk bwd, which raises NotImplementedError --
      # because idx_inv is the FULL inverse permutation, indexed by buffer position.
      padded_idx_inv = jnp.pad(idx_inv, (0, buffer_size))
      sliced_idx_inv = jax.lax.dynamic_slice_in_dim(padded_idx_inv, shard_output_start, buffer_size, axis=0)
      gather_end = jnp.minimum(shard_output_end - shard_output_start, buffer_size)
      padded_weights = jnp.pad(weight_for_sorted, (0, buffer_size))
      sliced_weights = jax.lax.dynamic_slice_in_dim(padded_weights, shard_output_start, buffer_size, axis=0)
      grad_sorted_tokens = ragged_gather(
          g_full,
          _bwd_ag_row(sliced_idx_inv // topk),
          jnp.int32(0)[None],
          gather_end[None],
          weights=sliced_weights,
          has_weights=True,
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )
    # None cotangents for group_sizes / ridx / w: no gradient flows to the routing weights
    # through the combine, matching the un-chunked ring_ragged_unsort path.
    return grad_sorted_tokens, None, None, None

  _chunked_combine_rs.defvjp(_chunked_combine_rs_fwd, _chunked_combine_rs_bwd)

  return _chunked_combine_rs(sorted_tokens_local, group_sizes_local, ridx, w)


def chunked_ring_dispatch(
    hidden_states_local,
    topk_indices_global,
    num_experts,
    topk,
    ep_name,
    ep_size,
    n_chunks,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    gather_flops_override=-1,
    gather_reduce_flops_override=-1,
    gather_bytes_accessed_override=-1,
    gather_reduce_bytes_accessed_override=-1,
):
  """Decoupled chunked dispatch (INPUT side): chunk the EP all-gather + ragged-sort over the
  INPUT-token axis so each chunk's all-gather (async ICI) hides under the PREVIOUS chunk's
  ragged-sort (SparseCore). The GMM runs FULL/un-chunked downstream.

  COMPACTION-FIRST FORWARD (Route B, rung 9e). The rung-9 flat-buffer variant merged each
  chunk into the sorted buffer via ``jnp.where(mask_c, gathered, buf)`` = N x full-buffer
  gather scratch + N x full-buffer TC merge traffic (measured 19.1 s/step vs 15.23 control);
  the in-place SC accumulate that would fix it is unshippable (no output aliasing in
  ``pl.kernel`` -- see ragged_gather_reduce_accumulate). Route B removes the merge entirely by
  DEFERRING buffer materialization:
    1. Per chunk c: AG(chunk c) -> ONE COMPACTED ``ragged_gather`` emitting ONLY chunk-c's
       buffer rows, in buffer order, into a small ``[buffer_size // n_chunks, hidden]`` piece.
       Every input token contributes exactly ``topk`` buffer rows, so each chunk owns EXACTLY
       ``buffer_size / n_chunks`` rows globally -- the piece size is static, no dynamic counts.
    2. Barrier chain (rung-6b fusion discipline): chunk c's gather inputs are fenced on piece
       c-1, so the SC gathers stay emission-ordered while AG(c+1) issues underneath.
    3. After the loop: concat the N pieces (``[buffer_size, hidden]`` total) + ONE full-buffer
       un-chunked-STYLE ``ragged_gather`` that places every row at its buffer position
       (``buffer[j] = pieces[piece_pos_of_row[j]]``, output row j == buffer position j, full
       [shard_output_start, shard_output_end) bounds AS-IS -- the rung-8 SAFE call class).
  SC cost: compaction (valid rows split across N chunks) + placement (same valid rows once)
  ~= 2x the un-chunked dispatch SC pass, traded for hiding the token-AG; the N x full-buffer
  merge traffic and the N live full-buffer scratches are GONE (pieces total ONE buffer).

  KERNEL-CONTRACT NOTES (the campaign's bug classes, handled explicitly):
    * The COMPACTION call has chunk-sized index/OUTPUT axes (consistent with each other, so
      the kernel's block/partition counts are self-consistent) and its [start, end) bounds are
      in the OUTPUT-ROW space of the call = the RANK of the buffer row within the chunk's
      ascending row list -- converted from the shard's buffer-position range via searchsorted
      (the rung-7/-8 rank-space rule; buffer positions passed raw would index chunk-sized
      arrays out of bounds). Its source ``xg_c`` is only random-accessed by index value; every
      ``chunk_local_row`` value is ``< per*ep_size == xg_c.shape[0]`` by construction.
    * The PLACEMENT call is the safe un-chunked shape class: full-length indices/output
      (n == buffer_size), so output row j IS buffer position j and the shard's
      buffer-position bounds are correct AS-IS. For j in [start, end) it reads piece position
      ``piece_pos_of_row[j]``, which lies inside the owning chunk's written rank window by
      construction -- unwritten (uninitialized-HBM) piece rows are never read.
    * Rows of the RESULT buffer outside [start, end) are unwritten, exactly like the
      un-chunked ``ring_ragged_sort`` gather (same call class/bounds); downstream reads only
      the shard's expert range.

  Buffer order is preserved EXACTLY (GLOBAL shard-major expert argsort, UNPERMUTED): the
  chunk-major reindex applies ONLY to which input rows each all-gather grabs, never to the
  buffer layout, so the GMM / downstream combine contract is unchanged (no vLLM token-permute).

  SINGLE custom_vjp, un-chunked backward (rung-6 Step-1 pattern, N-independent HBM): the M
  chunks are a partition of the un-chunked ``ring_ragged_sort``, so the backward is its
  un-chunked backward applied ONCE to the full cotangent -- one ``ragged_gather_reduce`` over
  ``topk_argsort_revert_indices`` (grad wrt the GLOBAL all-gathered x) then one ``psum_scatter``
  (the all-gather transpose -> grad wrt LOCAL x). Bit-exact vs the un-chunked dispatch backward.

  Args:
    hidden_states_local: 2D ``[num_tokens_local, hidden]`` PRE-all-gather local activations.
    topk_indices_global: 2D ``[num_tokens_global, topk]`` GLOBAL routed expert ids (routing is
      replicated: logits are all-gathered upstream, exactly as the un-chunked path).
    n_chunks: number of input-token chunks (M); must divide ``num_tokens_local``.

  Returns:
    ``(buffer, group_sizes_local, topk_argsort_revert_indices)`` matching ``ring_ragged_sort``
    (buffer is ``[num_tokens_global*topk, hidden]``, expert-sorted, masked to this shard's
    expert range).
  """
  num_tokens_local = hidden_states_local.shape[0]
  num_tokens_global = topk_indices_global.shape[0]
  hidden = hidden_states_local.shape[-1]
  assert num_tokens_global == num_tokens_local * ep_size, (
      f"chunked_ring_dispatch expects pre-AG local x: num_tokens_local={num_tokens_local}, "
      f"ep_size={ep_size}, num_tokens_global={num_tokens_global} (need global == local*ep)."
  )
  per = num_tokens_local // n_chunks  # local tokens per chunk (per-shard AG payload per chunk)
  assert per * n_chunks == num_tokens_local, (
      f"decouple_dispatch_chunks={n_chunks} must divide num_tokens_local={num_tokens_local}."
  )
  buffer_size = num_tokens_global * topk

  # --- Routing (index-only, replicated across shards): mirror ring_ragged_sort's forward. ---
  topk_indices_flat = topk_indices_global.flatten()  # [num_tokens_global * topk]
  topk_argsort_indices = jnp.argsort(topk_indices_flat)
  token_indices = jnp.arange(num_tokens_global, dtype=jnp.int32).repeat(topk)
  # buffer pos -> GLOBAL (shard-major) token id (UNPERMUTED -- preserves un-chunked buffer order).
  token_indices_sorted = token_indices[topk_argsort_indices]
  group_sizes_local = jax.nn.one_hot(topk_indices_flat, num_experts, dtype=jnp.int32).sum(axis=0)
  topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)

  shard_idx = jax.lax.axis_index(ep_name)
  local_num_experts = num_experts // ep_size
  experts_start = shard_idx * local_num_experts
  experts_end = experts_start + local_num_experts
  group_offsets = jnp.cumulative_sum(group_sizes_local, include_initial=True)
  shard_output_start = group_offsets[experts_start]
  shard_output_end = group_offsets[experts_end]

  # Per buffer row, decompose its source GLOBAL token into (origin shard, local idx, chunk,
  # in-chunk row). chunk c's all-gather array is shard-major within the chunk:
  #   xg_c row(g) = origin_shard * per + (local_idx - c*per)
  origin_shard = token_indices_sorted // num_tokens_local
  local_idx = token_indices_sorted % num_tokens_local
  chunk_of_row = local_idx // per  # which chunk owns this buffer row's token
  in_chunk_pos = local_idx - chunk_of_row * per
  chunk_local_row = origin_shard * per + in_chunk_pos  # row within the chunk's all-gathered array

  # Route-B compaction indexing (index-only, replicated). Stable argsort groups the buffer rows
  # by owning chunk, PRESERVING buffer order within each chunk; each chunk owns exactly
  # rows_per_chunk rows globally, so static slices of `order` are exact per-chunk row lists.
  rows_per_chunk = buffer_size // n_chunks
  order = jnp.argsort(chunk_of_row).astype(jnp.int32)  # piece position p -> buffer row (stable)
  piece_pos_of_row = jnp.argsort(order).astype(jnp.int32)  # buffer row j -> piece position p
  src_idx_all = chunk_local_row[order]  # piece position p -> source row in its chunk's xg_c

  # The tracer-derived index arrays are passed as EXPLICIT custom_vjp inputs, NOT closed over: a
  # custom_vjp closing over scan-carried TRACERS hits "No constant handler for DynamicJaxprTracer"
  # inside the per-layer scan (they become constvars). Mirrors chunked_ring_combine_reduce_scatter.
  # All are non-differentiable index/int arrays -> None cotangents; only x carries a gradient.
  @jax.custom_vjp
  def _chunked_dispatch(hidden_states_local, order, src_idx_all, piece_pos_of_row, s_start, s_end, revert):
    return _chunked_dispatch_fwd(hidden_states_local, order, src_idx_all, piece_pos_of_row, s_start, s_end, revert)[0]

  @jax.named_scope("chunked-dispatch-fwd")
  def _chunked_dispatch_fwd(hidden_states_local, order, src_idx_all, piece_pos_of_row, s_start, s_end, revert):
    # FORWARD (Route B): per-chunk AG + COMPACTED gather into a [rows_per_chunk] piece, then one
    # full-buffer placement gather. See the function docstring for the design + kernel contracts.
    x_chunks = hidden_states_local.reshape(n_chunks, per, hidden)
    pieces = []
    prev_piece = None
    for c in range(n_chunks):
      xg_c = jax.lax.all_gather(x_chunks[c], ep_name, axis=0, tiled=True)
      rows_c = order[c * rows_per_chunk : (c + 1) * rows_per_chunk]  # chunk-c buffer rows, ascending
      src_c = src_idx_all[c * rows_per_chunk : (c + 1) * rows_per_chunk]
      # RANK-space bounds (rung-7/-8 rule): the compaction call's output rows are the RANKS of
      # chunk-c's buffer rows, so the shard's buffer-position range must be converted via
      # searchsorted over the ascending per-chunk row list.
      cs = jnp.searchsorted(rows_c, s_start).astype(jnp.int32)
      ce = jnp.searchsorted(rows_c, s_end).astype(jnp.int32)
      if prev_piece is not None:
        # Barrier chain (rung-6b): fence this chunk's GATHER inputs on the previous piece so the
        # SC compactions stay emission-ordered and cannot fuse across chunks, while AG(c) itself
        # (upstream of this fence) issues early and overlaps compaction(c-1).
        xg_c, src_c, _ = jax.lax.optimization_barrier((xg_c, src_c, prev_piece))
      piece = ragged_gather(
          xg_c,
          src_c,
          cs[None],
          ce[None],
          enforce_fallback=enforce_gather_fallback,
          flops_override=gather_flops_override,
          bytes_accessed_override=gather_bytes_accessed_override,
      )
      # Fence the piece so no downstream fusion spans chunks (mirrors the combine-side chain).
      piece = jax.lax.optimization_barrier(piece)
      prev_piece = piece
      pieces.append(piece)
    pieces_all = jnp.concatenate(pieces, axis=0)  # [buffer_size, hidden], chunk-grouped rows
    # PLACEMENT: un-chunked-style call (output row j == buffer position j; bounds AS-IS). For
    # j in [s_start, s_end), piece_pos_of_row[j] lies inside its chunk's written rank window.
    buf = ragged_gather(
        pieces_all,
        piece_pos_of_row,
        s_start[None],
        s_end[None],
        enforce_fallback=enforce_gather_fallback,
        flops_override=gather_flops_override,
        bytes_accessed_override=gather_bytes_accessed_override,
    )
    # Residual: index-only tracers + local shape. NOTHING heavy (no buffer), mirroring
    # ring_ragged_sort's bwd which recomputes grad from g_out + indices alone.
    return buf, (s_start, s_end, revert, hidden_states_local.shape)

  @jax.named_scope("chunked-dispatch-bwd")
  def _chunked_dispatch_bwd(res, g_out):
    # BACKWARD: the M chunks partition the un-chunked ring_ragged_sort, so its backward applied
    # ONCE to the full g_out -> grad wrt GLOBAL x, then the all-gather transpose (psum_scatter)
    # -> grad wrt LOCAL x. One gather-reduce + one RS, FLAT in n_chunks, bit-exact vs un-chunked.
    s_start, s_end, revert, local_shape = res
    n = revert.shape[0]
    valid_rows_mask = (revert >= s_start) & (revert < s_end)
    grad_global = ragged_gather_reduce(
        g_out,
        revert,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=valid_rows_mask,
        reduce_group_size=topk,
        enforce_fallback=enforce_gather_reduce_fallback,
        flops_override=gather_reduce_flops_override,
        bytes_accessed_override=gather_reduce_bytes_accessed_override,
    )  # [num_tokens_global, hidden] -- grad wrt the ALL-GATHERED x
    # Transpose of the forward all_gather(tiled) is a tiled psum_scatter over the same axis.
    grad_local = jax.lax.psum_scatter(grad_global, ep_name, scatter_dimension=0, tiled=True)
    grad_local = grad_local.reshape(local_shape).astype(g_out.dtype)
    return (grad_local, None, None, None, None, None, None)

  _chunked_dispatch.defvjp(_chunked_dispatch_fwd, _chunked_dispatch_bwd)

  buffer = _chunked_dispatch(
      hidden_states_local,
      order,
      src_idx_all,
      piece_pos_of_row,
      shard_output_start,
      shard_output_end,
      topk_argsort_revert_indices,
  )
  return buffer, group_sizes_local, topk_argsort_revert_indices


def a2a_ragged_sort(
    inputs,
    sort_indices,
    valid_end,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    use_single_sparsecore=False,
):
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
    out = ragged_gather(
        inputs,
        sort_indices,
        start[None],
        end[None],
        use_single_sparsecore=use_single_sparsecore,
    )
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
        use_single_sparsecore=use_single_sparsecore,
    )
    # custom_vjp must return one gradient per primal arg; valid_end is integer
    # and non-differentiable, so we return None for it.
    return grad_inputs, None, None

  _a2a_ragged_sort.defvjp(_a2a_ragged_sort_fwd, _a2a_ragged_sort_bwd)
  return _a2a_ragged_sort(inputs, sort_indices, valid_end)


def a2a_ragged_unsort(
    sorted_tokens,
    revert_indices,
    valid_end,
    enforce_gather_fallback=False,
    enforce_gather_reduce_fallback=False,
    use_single_sparsecore=False,
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
        use_single_sparsecore=use_single_sparsecore,
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
    grad_sorted = ragged_gather(
        g_out,
        idx_inv,
        start[None],
        end[None],
        use_single_sparsecore=use_single_sparsecore,
    )
    num_rows = sorted_tokens_shape[0]
    pos = jnp.arange(num_rows)
    valid = pos < end
    grad_sorted = jnp.where(valid[:, None], grad_sorted, jnp.zeros_like(grad_sorted))

    return grad_sorted, None, None

  _a2a_ragged_unsort.defvjp(_a2a_ragged_unsort_fwd, _a2a_ragged_unsort_bwd)
  return _a2a_ragged_unsort(sorted_tokens, revert_indices, valid_end)
