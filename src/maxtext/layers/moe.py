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


"""MoE related Layers."""

import enum
import functools
import math
import random
from typing import Iterable, Optional, Tuple, Union

from aqt.jax.v2 import aqt_tensor as aqt
from flax import nnx
from flax import struct
import jax
from jax import ad_checkpoint as adc
from jax.experimental import xla_metadata
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.common import common_types as ctypes
from maxtext.common.common_types import ShardMode
from maxtext.kernels import megablox as mblx
from maxtext.layers import attentions, linears, nnx_wrappers, quantizations
from maxtext.layers.initializers import NdInitializer, default_bias_init, nd_dense_init, variable_to_logically_partitioned
from maxtext.kernels.ragged.ragged_sort import a2a_ragged_sort
from maxtext.kernels.ragged.ragged_sort import a2a_ragged_unsort
from maxtext.kernels.ragged.ragged_sort import chunked_ring_combine_reduce_scatter
from maxtext.kernels.ragged.ragged_sort import chunked_ring_dispatch
from maxtext.kernels.ragged.ragged_sort import compute_ring_sort_indices
from maxtext.kernels.ragged.ragged_sort import ring_ragged_sort
from maxtext.kernels.ragged.ragged_sort import ring_ragged_unsort
from maxtext.kernels.ring_ag import ring_all_gather
from maxtext.kernels.ring_ag import ring_reduce_scatter
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import create_sharding, maybe_shard_with_logical, maybe_shard_with_pspec
from maxtext.utils.sharding import logical_to_mesh_axes, remove_expert_from_partition_spec, get_logical_axis_rules
import numpy as np
import qwix
from qwix.contrib.sparsity import sparsity_module
import qwix.pallas as qpl
import tokamax


@jax.custom_vjp
def _ste_quant(w, sc):
  """Straight-through e4m3 quantizer for moe_fp8_boundary_qag.

  Forward: e4m3 qvalue = clip(w / sc, +-448). Backward (STE): pass the incoming cotangent straight
  through to w in BF16 UNCHANGED. The gmm backward computes drhs with NO rhs.scale applied (it is
  already dL/d(dequantized weight)), and w -> qv*sc is the identity chain, so the STE vjp is a plain
  pass-through -- a /sc here would inflate the per-channel gradient by 1/sc. This also keeps the
  weight gradient in bf16 (never e4m3 -- an e4m3 gradient overflows and NaNs the reduce-scatter).
  """
  return jnp.clip(w / sc.astype(w.dtype), -448.0, 448.0).astype(jnp.float8_e4m3fn)


def _ste_quant_fwd(w, sc):
  return _ste_quant(w, sc), None


def _ste_quant_bwd(_res, g):
  return (g.astype(jnp.bfloat16), None)


_ste_quant.defvjp(_ste_quant_fwd, _ste_quant_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _ring_ct_reduce_scatter(output, mesh, ep_name, collective_id):
  """The combine reduce-scatter with its backward cotangent all-gather on the TC RING kernel.

  FORWARD: byte-identical to the stock path -- the plain XLA
  ``jax.lax.psum_scatter(output, ep_name, scatter_dimension=0, tiled=True)`` (also what the remat
  recompute re-traces: the primal is the plain collective, so NO Pallas DMA ever runs in a
  rematted region). BACKWARD: the autodiff transpose of that tiled psum_scatter is a tiled EP
  all-gather of the loop-carried cotangent (bf16 [tokens_local, ...] -> [num_tokens, ...]) -- the
  #1+#2 worst-overlap collectives in the record profile (~810ms/step pair): as an XLA collective
  it rides the SparseCore all-gather-offload queue and stalls behind the SC combines. Here it runs
  on the bidirectional store-and-forward TC ring kernel instead (ICI DMAs on the TensorCore, where
  the backward has slack), numerically == lax.all_gather (pure tiled data move, bit-exact).
  """
  return jax.lax.psum_scatter(output, ep_name, scatter_dimension=0, tiled=True)


def _ring_ct_rs_fwd(output, mesh, ep_name, collective_id):
  return _ring_ct_reduce_scatter(output, mesh, ep_name, collective_id), None


def _ring_ct_rs_bwd(mesh, ep_name, collective_id, _res, ct):
  return (ring_all_gather(ct, mesh, (ep_name,), 0, collective_id),)


_ring_ct_reduce_scatter.defvjp(_ring_ct_rs_fwd, _ring_ct_rs_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _ring_combine_rs(output, mesh, ep_name, rs_collective_id, ag_collective_id):
  """The combine reduce-scatter with BOTH directions on TC ring Pallas kernels
  (moe_ring_combine_rs, composes with/requires moe_ring_cotangent_ag).

  FORWARD: the bidirectional ring reduce-scatter kernel (== psum_scatter, rel=0 validated; per-hop
  f32-staged add like the XLA ring, but the hop ORDER may differ -> bf16 reduce-order noise, not
  bit-exact). BACKWARD: the ring all-gather on the cotangent (as _ring_ct_reduce_scatter).
  REMAT: no save is needed -- the dump census shows the combine is not part of the backward remat
  recompute (it stops at the mlpwo/moe_mlpwo checkpoint save upstream), so the forward Pallas DMA
  never re-runs in a rematted region.
  """
  return ring_reduce_scatter(output, mesh, ep_name, 0, rs_collective_id)


def _ring_combine_rs_fwd(output, mesh, ep_name, rs_collective_id, ag_collective_id):
  return _ring_combine_rs(output, mesh, ep_name, rs_collective_id, ag_collective_id), None


def _ring_combine_rs_bwd(mesh, ep_name, rs_collective_id, ag_collective_id, _res, ct):
  return (ring_all_gather(ct, mesh, (ep_name,), 0, ag_collective_id),)


_ring_combine_rs.defvjp(_ring_combine_rs_fwd, _ring_combine_rs_bwd)

set_xla_metadata = xla_metadata.set_xla_metadata


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _direct_reduce_scatter(output, mesh, ep_name, collective_id, sched_group=None):
  """Direct-to-owner Pallas reduce-scatter over the EP axis -- a drop-in for
  `jax.lax.psum_scatter(output, ep_name, scatter_dimension=0, tiled=True)`.

  Each device sends its chunk-c straight to owner c (pure async ICI DMA on the
  TensorCore), then a local dense f32 sum reduces. Because it is a TC Pallas kernel firing
  async ICI copies -- not an XLA collective -- it (a) does not conflict with the SparseCore
  offload queue that serializes psum_scatter behind the SC combines, and (b) is immune to the
  v7x prohibition on async-RS continuation fusion. Ported from the fused-combine-rs campaign
  (commits b5bfd57b2 + a1a7f2179: verified == psum_scatter in isolation, rel 0.004 bf16;
  prototype provenance perf-drills/gather/combine/{direct_rs,verify_combine_rs}.py, v5p).
  MUST be called INSIDE the MoE shard_map so the ambient mesh axes are available to
  `lax.axis_index`. `collective_id` selects the barrier semaphore: concurrent in-flight
  instances (the per-chunk RSs of decouple_combine_rs_chunks may overlap in the schedule)
  must each use a DISTINCT id or their entry barriers would count each other's signals.

  `sched_group` (None by default) is a pure-scheduling hint applied ONLY in the BACKWARD: when
  not None the transpose all-gather (`_drs_bwd`) is tagged with that XLA `_scheduling_group_id`
  so the latency-hiding scheduler treats the (exposed) combine cotangent all-gather as an overlap
  candidate with other same-group ops (the splash host-offload restore copies -- see
  moe_splash_offload_scheduling_group). The forward is unaffected; None => byte-identical.
  """
  ep_size = mesh.shape[ep_name]
  axis_names = mesh.axis_names
  mesh_shape = mesh.shape
  n = output.shape[0]
  chunk = n // ep_size
  trailing = tuple(output.shape[1:])

  def _mesh_device_id(ep_rank):
    # Full mesh-coordinate tuple in mesh.axis_names ORDER (DeviceIdType.MESH resolves it
    # against the device mesh): `ep_rank` in the expert slot, every other axis pinned to its
    # current axis_index (0 for size-1 axes -- avoid a needless axis_index call). The order
    # MUST match the mesh or the DMA targets the wrong device.
    return tuple(
        ep_rank if nm == ep_name else (0 if mesh_shape[nm] == 1 else jax.lax.axis_index(nm)) for nm in axis_names
    )

  def _kern(y_ref, o_ref, send, recv):
    my = jax.lax.axis_index(ep_name)
    # Full EP-group barrier (collective_id on the pallas_call enables the barrier sem).
    bsem = pltpu.get_barrier_semaphore()
    for c in range(ep_size):
      pltpu.semaphore_signal(bsem, inc=1, device_id=_mesh_device_id(c), device_id_type=pl.DeviceIdType.MESH)
    pltpu.semaphore_wait(bsem, ep_size)
    sends = []
    for c in range(ep_size):  # scatter my chunk-c -> owner c's recv[my]
      cp = pltpu.make_async_remote_copy(
          y_ref.at[pl.ds(c * chunk, chunk)],
          o_ref.at[my],
          send.at[c],
          recv.at[my],
          device_id=_mesh_device_id(c),
          device_id_type=pl.DeviceIdType.MESH,
      )
      cp.start()
      sends.append(cp)
    for d in range(ep_size):  # wait for chunk-(my) arriving from each device d -> recv[d]
      pltpu.make_async_remote_copy(
          y_ref.at[pl.ds(0, chunk)],
          o_ref.at[d],
          send.at[d],
          recv.at[d],
          device_id=_mesh_device_id(d),
          device_id_type=pl.DeviceIdType.MESH,
      ).wait_recv()
    for cp in sends:
      cp.wait_send()

  recv = pl.pallas_call(
      _kern,
      out_shape=jax.ShapeDtypeStruct((ep_size, chunk) + trailing, output.dtype),
      in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
      out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
      scratch_shapes=[pltpu.SemaphoreType.DMA((ep_size,)), pltpu.SemaphoreType.DMA((ep_size,))],
      compiler_params=pltpu.CompilerParams(collective_id=collective_id),
  )(output)
  return recv.astype(jnp.float32).sum(0).astype(output.dtype)


# The Pallas kernel is opaque to autodiff (no jvp). Give it the SAME differentiation as the
# `psum_scatter` it replaces: the transpose of a tiled reduce-scatter over EP (scatter_dim=0)
# is a tiled all-gather over EP. Forward values are verified == psum_scatter, so fwd+bwd match.
def _drs_fwd(output, mesh, ep_name, collective_id, sched_group=None):
  return _direct_reduce_scatter(output, mesh, ep_name, collective_id, sched_group), None


def _drs_bwd(mesh, ep_name, collective_id, sched_group, _res, ct):
  # sched_group (moe_splash_offload_scheduling_group): tag this combine cotangent all-gather
  # (== all-gather.626, the transpose of the direct RS) so the scheduler can overlap the ICI AG
  # with the host->device splash restore copies tagged into the same group. None => untagged
  # (byte-identical; only a frontend attribute is added, dataflow/numerics are unchanged).
  if sched_group is None:
    return (jax.lax.all_gather(ct, ep_name, axis=0, tiled=True),)
  with _scheduling_group(sched_group):
    return (jax.lax.all_gather(ct, ep_name, axis=0, tiled=True),)


_direct_reduce_scatter.defvjp(_drs_fwd, _drs_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _direct_all_gather(x, mesh, ep_name, collective_id):
  """Direct-to-owner Pallas all-gather over the EP axis -- a drop-in for
  `jax.lax.all_gather(x, axis_name=ep_name, axis=0, tiled=True)`.

  Symmetric counterpart of `_direct_reduce_scatter`: each device broadcasts its local shard to
  slot `my` in every EP peer's output buffer (pure async ICI DMA on the TensorCore), and receives
  every peer d's shard into slot d; the [ep_size, chunk, ...] receive buffer is then reshaped to
  the [ep_size*chunk, ...] concatenation lax.all_gather(tiled=True, axis=0) produces. Because it is
  a TC Pallas kernel firing async ICI copies -- NOT an XLA collective -- it (a) does NOT ride the
  SparseCore offload queue that (with the single-SC-for-all-gather-offload path) serializes the EP
  backward EP all-gather (either the token/activation gather -- moe_direct_token_ag -- or the
  combine-cotangent all-gather == all-gather.626, the transpose of the direct-RS -- moe_direct_combine_ag)
  behind the SC-resident weight re-gather / combines in the backward, so XLA can overlap its ICI DMAs
  with that SC work (different engines), and (b) is immune to the async-collective continuation-fusion
  restrictions the psum_scatter/all_gather collectives hit. Prototype provenance:
  perf-drills/gather/weight_ag.py (TC-AG ∥ SC-gather MECHANISM proof, v5p: correctness ==
  lax.all_gather + measured overlap where the XLA collective did not). Shared by both direct-AG
  call sites; each passes its own DISTINCT collective_id (40 token / 50 combine) so two in-flight
  instances never share an entry-barrier semaphore.

  MUST be called INSIDE the MoE shard_map so the ambient mesh axes are available to
  `lax.axis_index`. `collective_id` selects the barrier semaphore: it must be DISTINCT from every
  concurrently in-flight direct-RS id (the RS uses 7..7+chunks) so an in-flight RS and AG never
  count each other's entry-barrier signals. Single-axis EP only (ep_name a str); the caller falls
  back to lax.all_gather otherwise.
  """
  ep_size = mesh.shape[ep_name]
  axis_names = mesh.axis_names
  mesh_shape = mesh.shape
  chunk = x.shape[0]
  trailing = tuple(x.shape[1:])

  def _mesh_device_id(ep_rank):
    # Full mesh-coordinate tuple in mesh.axis_names ORDER (identical convention to the direct-RS):
    # `ep_rank` in the expert slot, every other axis pinned to its current axis_index.
    return tuple(
        ep_rank if nm == ep_name else (0 if mesh_shape[nm] == 1 else jax.lax.axis_index(nm)) for nm in axis_names
    )

  def _kern(x_ref, o_ref, send, recv):
    my = jax.lax.axis_index(ep_name)
    # Full EP-group barrier (collective_id on the pallas_call enables the barrier sem).
    bsem = pltpu.get_barrier_semaphore()
    for c in range(ep_size):
      pltpu.semaphore_signal(bsem, inc=1, device_id=_mesh_device_id(c), device_id_type=pl.DeviceIdType.MESH)
    pltpu.semaphore_wait(bsem, ep_size)
    sends = []
    for c in range(ep_size):  # broadcast my shard -> slot `my` in every peer c's buffer
      cp = pltpu.make_async_remote_copy(
          x_ref,
          o_ref.at[my],
          send.at[c],
          recv.at[my],
          device_id=_mesh_device_id(c),
          device_id_type=pl.DeviceIdType.MESH,
      )
      cp.start()
      sends.append(cp)
    for d in range(ep_size):  # receive peer d's shard into slot d
      pltpu.make_async_remote_copy(
          x_ref,
          o_ref.at[d],
          send.at[d],
          recv.at[d],
          device_id=_mesh_device_id(d),
          device_id_type=pl.DeviceIdType.MESH,
      ).wait_recv()
    for cp in sends:
      cp.wait_send()

  recv = pl.pallas_call(
      _kern,
      out_shape=jax.ShapeDtypeStruct((ep_size, chunk) + trailing, x.dtype),
      in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
      out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
      scratch_shapes=[pltpu.SemaphoreType.DMA((ep_size,)), pltpu.SemaphoreType.DMA((ep_size,))],
      compiler_params=pltpu.CompilerParams(collective_id=collective_id),
  )(x)
  return recv.reshape((ep_size * chunk,) + trailing)


# The Pallas kernel is opaque to autodiff. Give it the SAME differentiation as the `lax.all_gather`
# it replaces: the transpose of a tiled all-gather over EP (scatter/gather dim 0) is a tiled
# reduce-scatter over EP. Forward values are verified == lax.all_gather, so fwd+bwd match. The bwd
# is PURE XLA (psum_scatter) -- no SC Pallas in the custom_vjp bwd, so no "No constant handler" wall.
def _dag_fwd(x, mesh, ep_name, collective_id):
  return _direct_all_gather(x, mesh, ep_name, collective_id), None


def _dag_bwd(mesh, ep_name, collective_id, _res, ct):
  return (jax.lax.psum_scatter(ct, ep_name, scatter_dimension=0, tiled=True),)


_direct_all_gather.defvjp(_dag_fwd, _dag_bwd)

# Barrier-semaphore collective_ids for the two BACKWARD direct all-gather call sites. Both are held
# clear of the direct-RS ids (7..7+decouple_combine_rs_chunks, realistically <=~23) so an in-flight
# RS and either AG never share an entry barrier, and they are DISTINCT from EACH OTHER (40 vs 50) so
# a token-AG and a combine-AG that happen to be in flight together never count each other's barrier
# signals. (In practice token-AG is dispatch-phase and combine-AG is combine-phase, so they are not
# concurrently in flight anyway; distinct ids are belt-and-suspenders.)
_DIRECT_TOKEN_AG_COLLECTIVE_ID = 40  # moe_direct_token_ag: backward-recompute EP token all-gather
_DIRECT_FWD_TOKEN_AG_COLLECTIVE_ID = 45  # moe_fwd_direct_token_ag: FORWARD EP token dispatch all-gather
_DIRECT_COMBINE_AG_COLLECTIVE_ID = 50  # moe_direct_combine_ag: backward combine-cotangent all-gather (== .626)
_RING_CT_AG_COLLECTIVE_ID = 55  # moe_ring_cotangent_ag: backward combine-cotangent RING all-gather
_RING_RS_COLLECTIVE_ID = 56  # moe_ring_combine_rs: FORWARD combine RING reduce-scatter


def _scheduling_group(group_id):
  """Tag enclosed ops with an XLA `_scheduling_group_id`.

  Instructions sharing a `_scheduling_group_id` are candidates for the XLA
  scheduler to overlap (see batchsplit's `scheduling_group`). Used here to tag
  the explicit FSDP weight all-gather so the scheduler overlaps the (otherwise
  exposed) weight-AG with attention-phase compute in the same decoder layer.
  """
  return set_xla_metadata(_scheduling_group_id=group_id)


# Fixed scheduling-group id shared by the MoE FSDP weight all-gather and the
# attention earlier in the same (scanned) decoder layer. Under
# scan_layers=true the body is traced once, so a fixed id scopes to one layer.
_WEIGHT_AG_SCHED_GROUP = 1

# Scheduling-group id shared, in the BACKWARD only, by the combine cotangent all-gather
# (_drs_bwd, == all-gather.626) and the splash host-offload (context, lse) restore copies
# (deepseek._attn_host), so the latency-hiding scheduler overlaps the exposed ICI AG with the
# otherwise-idle Host-DMA restore lane. Gated on moe_splash_offload_scheduling_group. Distinct
# from the forward weight-AG groups (1..3) and the observed splash/attention groups so the
# all-gather-combiner cannot fuse it into an un-hideable monolith with them.
_SPLASH_OFFLOAD_SCHED_GROUP = 30


DISPATCH = "dispatch"
COMBINE = "combine"


@struct.dataclass
class RouteMetadata:
  """EP communication state needed to undo the forward all-to-all after expert computation."""

  # Index of this device's EP shard.
  expert_shard_id: int
  # Permutation of [tokens received by this expert shard], sorted by local expert ID.
  local_sorted_indices: Optional[jax.Array]
  # Shape [num_ep].  Aggregates group_sizes per EP shard; tracks how many local tokens are routed to each EP shard.
  reshaped_group_sizes: Optional[jax.Array]
  # Shape [num_ep, num_ep]. all_gather of reshaped_group_sizes across EP shards.
  # [i, j] = number of tokens from batch shard i sent to expert shard j.
  all_shards_group_sizes: Optional[jax.Array]


@struct.dataclass
class RouteOutput:
  """Holds state of routing output"""

  # Shape [num experts], tracks number of local tokens routed to every expert.
  group_sizes: jax.Array
  # Indices of experts chosen for each token.
  selected_experts: jax.Array
  # Tokens sorted by experts they are routed to.
  sorted_selected_experts: jax.Array
  # Weights for each of the selected experts.
  weights: jax.Array
  # Auxiliary loss for token distribution among experts.
  lb_loss: Optional[jax.Array]
  # Dynamic bias updates for loss-free load balancing, used only for Deepseek models
  bias_updates: Optional[jax.Array]
  # Shape [local experts], tracks number of local tokens routed to every local expert.
  local_group_sizes: Optional[jax.Array] = None


def _truncate_matrix(all_shards_group_sizes: jax.Array, buffer_size: int) -> jax.Array:
  """Truncates the traffic matrix to fit in buffer_size on receiver side.

  When ragged_buffer_factor > 0, the receiver buffer has a fixed capacity
  (buffer_size). Due to routing imbalance, some shards might receive more tokens
  than this capacity. We use a prefix sum to deterministically truncate the
  received tokens on all shards, ensuring we don't write out of bounds.
  """
  cumsum = jnp.cumsum(all_shards_group_sizes, axis=0)
  clamped_cumsum = jnp.minimum(cumsum, buffer_size)
  clamped_cumsum_extended = jnp.concatenate(
      [
          jnp.zeros((1, all_shards_group_sizes.shape[1]), dtype=clamped_cumsum.dtype),
          clamped_cumsum,
      ],
      axis=0,
  )
  return jnp.diff(clamped_cumsum_extended, axis=0)


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
    use_custom_vjp: bool,
) -> jax.Array:
  """Sort activations by `sort_indices`.

  If `use_custom_vjp=True`, then we use a custom backward pass that
  reverses the sort order. Specifically, this unsort operation is simply a sort
  with `jnp.argsort(sort_indices)` as the sort indices. This is only needed in
  the case where the compiler generates a less efficient backward pass op.

  Note that `use_custom_vjp=True` assumes that `sort_indices` is a permutation
  of `jnp.arange(inputs.shape[0])`.

  Args:
    inputs: `(tokens, ...)`-shaped array of input activations to sort.
    sort_indices: `(tokens,)`-shaped array containing the sort order.
    use_custom_vjp: Whether to use the explicit backward pass.

  Returns:
    `(tokens, ...)`-shaped array of input activations sorted by `sort_indices`.
  """
  assert inputs.shape[0] == sort_indices.shape[0]

  with jax.named_scope("sort_activations"):
    if use_custom_vjp:
      return _sort_activations_custom(inputs, sort_indices)
    return inputs[sort_indices, ...]


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
  """Sort functions with custom vjp."""
  return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Forward pass of the custom vjp for `_sort_activations()`."""
  return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  """Backward pass of the custom vjp for `_sort_activations()`."""
  sort_indices = residuals
  return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def get_batchsplit_init_kernel_axes():
  return (
      ("expert_only", "embed_moe", None),
      ("expert_only", None, "embed_moe"),
  )


def random_routing(rng_key, gate_logits, num_experts_per_tok):
  """Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
      representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the indices of the selected experts for each
                       token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the weights for the selected experts.
  """
  bs, seq_len, num_experts = gate_logits.shape
  selected_num = bs * seq_len * num_experts_per_tok
  # Directly generate random integers in the range [0, num_experts)
  top_k_indices = jax.random.randint(
      rng_key,
      shape=(selected_num,),
      minval=0,
      maxval=num_experts,
      dtype=jnp.int32,
  )
  top_k_indices = top_k_indices.reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


def calculate_load_balance_updates(top_k_indices, num_experts, rate):
  """
  Computes a bias adjustment update based on expert load.
  Used in DeepSeek V3: https://arxiv.org/html/2412.19437v1.
  Implementation reference: https://arxiv.org/pdf/2408.15664.

  Args:
      top_k_indices: Shape (batch, sequence, top_k).
      num_experts: Total number of experts.
      rate: The update rate.

  Returns:
      update: The value to add to the expert bias. Shape (num_experts,).
  """
  flat_indices = top_k_indices.ravel()
  expert_counts = jnp.bincount(flat_indices, length=num_experts)

  total_tokens = flat_indices.size
  average_load = total_tokens / num_experts
  direction = jnp.sign(average_load - expert_counts)
  output = direction * rate
  return output


class Tid2EidVar(nnx.Variable):
  """Custom variable to hold tid2eid without trainable param overhead."""


class GateLogit(nnx.Module):
  """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing."""

  def __init__(
      self,
      in_features_shape: Union[Iterable[int], int],
      out_features_shape: Union[Iterable[int], int],
      model_name: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      axis: Union[Iterable[int], int] = -1,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      use_bias: bool = False,
      score_func: str = "",
      quant: Optional[quantizations.AqtQuantization] = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
  ):
    """Initializes the GateLogit module.

    Attributes:
      in_features_shape: The shape of the input features.
      out_features_shape: The shape of the output features, typically the number of experts.
      model_name: The name of the model.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      axis: The axis or axes over transformation is applied.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      use_bias: Whether to add learnable bias in gate logit scores. When enabled,
        this bias aids expert load balancing (like in DeepSeek V3), and is not
        part of the loss calculation.
      score_func: Scoring function for output normalization before applying bias.
      quant: The quantization configuration. If None, no quantization is applied.
      matmul_precision: The precision level for the matrix multiplication.
    """
    self.in_features_shape = linears.canonicalize_tuple(in_features_shape)
    self.out_features_shape = linears.canonicalize_tuple(out_features_shape)
    self.model_name = model_name
    self.mesh = mesh
    self.axis = linears.canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.use_bias = use_bias
    self.score_func = score_func
    self.quant = quant
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          out_sharding=bias_axes,
      )
    else:
      self.bias = None

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      dummy_inputs = jnp.zeros((1, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def __call__(self, inputs: jax.Array, _initializing: bool = False) -> Tuple[jax.Array, Optional[jax.Array]]:
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = linears.normalize_axes(self.axis, inputs.ndim)

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = self.kernel[...]
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(norm_axis)))
    output_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.shard_mode == ShardMode.EXPLICIT
        else None
    )
    output = linears._compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
        out_sharding=output_sharding,
    )
    pre_bias_logits = None

    if self.score_func:
      output = linears._convert_to_activation_function(self.score_func)(output)

    # NOTE: deepseek2 has a different pattern
    if self.model_name.startswith(("deepseek3", "deepseek4")):
      pre_bias_logits = output

    if self.use_bias:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nnx.Module):
  """Implements a routed MoE block."""

  def __init__(
      self,
      config: ctypes.Config,
      num_experts: int,
      num_experts_per_tok: int,
      mesh: jax.sharding.Mesh,
      kernel_init: attentions.NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      intermediate_dim: int = 2048,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
      is_hash_routing: bool = False,
  ):
    """Initializes the RoutedMoE module.

    Attributes:
      config: The main config setting.
      num_experts: Number of experts.
      num_experts_per_tok: Number of experts for each token.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      intermediate_dim: Intermediate dimension of MoE.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
      is_hash_routing: Whether this layer uses deterministic hash routing instead of top-K routing.
    """
    self.config = config
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.intermediate_dim = intermediate_dim
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    self.is_hash_routing = is_hash_routing

    # DeepSeek V4 Hash Routing
    if self.is_hash_routing:
      # Token-ID to Expert-ID lookup table for static routing
      # Must be stored as float32 because MaxText passes the entire variable tree
      # through jax.value_and_grad, which strictly requires all leaves to be inexact types
      # (even if they receive no gradients). We cast to int32 dynamically during routing.
      self.tid2eid = Tid2EidVar(
          jnp.zeros(
              (self.config.vocab_size, self.num_experts_per_tok),
              dtype=jnp.float32,
          ),
          out_sharding=None,  # Replicated across shards for local lookup
      )
    else:
      self.tid2eid = None

    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    if self.config.shard_exp_on_fsdp:
      # special sharding for dsv3
      self.wi_kernel_axes = ("embed_moe", None, "mlp_moe")
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_2d_fsdp_sharding:
      self.wi_kernel_axes = ("embed_moe", "mlp_moe", None)
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_batch_split_schedule:
      self.wi_kernel_axes, self.wo_kernel_axes = get_batchsplit_init_kernel_axes()
    else:
      self.wi_kernel_axes = ("exp", "embed_moe", "mlp_moe")
      self.wo_kernel_axes = ("exp", "mlp_moe", "embed_moe")

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa"):
      # vLLM uses 'model' as the tensor parallelism axis name
      self._tensor_parallelism_name = ("model", "attn_dp")
    else:
      self._tensor_parallelism_name = "tensor"

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") and self.config.enable_dp_attention:
      self._expert_parallelism_name = "attn_dp_expert"
    elif self.config.custom_mesh_and_rule == ctypes.CustomRule.CP_AS_EP:
      # when custom mesh and rule is cp-as-ep, context axis is same with expert in MoE component
      self._expert_parallelism_name = ("context", "expert")
    else:
      self._expert_parallelism_name = "expert"

    self.gate = GateLogit(
        in_features_shape=self.moe_expert_input_dim,
        out_features_shape=self.num_experts,
        mesh=self.mesh,
        model_name=self.config.model_name,
        dtype=jnp.float32 if self.config.float32_gate_logits else self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        use_bias=self.config.routed_bias and not self.is_hash_routing,
        # tpu-inference applies the score function in the fused_moe_gmm kernel,
        # so we don't apply it here to avoid redundant computation.
        # See https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/fused_moe_gmm.py#L58.
        score_func="" if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") else self.config.routed_score_func,
        matmul_precision=self.config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )
    rule = qpl.get_current_rule("gmm")
    sparsity_rule = None
    if rule is not None:
      if not isinstance(rule, qwix.QtRule):
        raise ValueError("Expect a QtRule for quantized training.")
      if rule.additional_qt_config and "sparsity_rule" in rule.additional_qt_config:
        q_s_rule = rule.additional_qt_config["sparsity_rule"]
        if q_s_rule and q_s_rule.weight_sparsity_n and q_s_rule.weight_sparsity_m:
          sparsity_rule = q_s_rule

    if sparsity_rule is not None:
      self.wi_0_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wi_1_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wo_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.intermediate_dim, self.config.emb_dim),
          sharding_axes=self.wo_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
    else:
      self.wi_0_sparsity_module = None
      self.wi_1_sparsity_module = None
      self.wo_sparsity_module = None

    # pylint: disable=protected-access
    self.activation_fn = linears._convert_to_activation_function(self.config.mlp_activations[0])

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)
    # Pad the MoE input weight kernels for GMM_v2 execution when configured.
    moe_intermediate_dim = (
        self.config.padded_base_moe_mlp_dim if self.config.padded_base_moe_mlp_dim is not None else self.intermediate_dim
    )

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save
      # memory. Instead they are retrieved from the tensors stored in the 'aqt'
      # collection.
      self.wi_0 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wi_1 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wo = jnp.zeros((num_experts, intermediate_dim, self.moe_expert_input_dim))
    elif self.config.prefuse_moe_weights:
      self.wi = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim * 2),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (
                  self.num_experts,
                  self.intermediate_dim,
                  self.moe_expert_input_dim,
              ),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )
    else:
      self.wi_0 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wi_1 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (
                  self.num_experts,
                  self.intermediate_dim,
                  self.moe_expert_input_dim,
              ),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )

    if self.config.mlp_bias:
      wi_bias_axes = ("exp", "activation_mlp")
      wo_bias_axes = ("exp", "activation_embed")
      wi_bias_shape = (self.num_experts, self.intermediate_dim)
      wo_bias_shape = (self.num_experts, self.moe_expert_input_dim)
      self.wi_0_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wi_1_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wo_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wo_bias_shape, self.weight_dtype),
          out_sharding=wo_bias_axes,
      )
    else:
      self.wi_0_bias = None
      self.wi_1_bias = None
      self.wo_bias = None

    if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      self.per_expert_scale = nnx.Param(
          jnp.ones((self.num_experts,), dtype=self.weight_dtype),
          out_sharding=("exp",),
      )
    else:
      self.per_expert_scale = None

    # Scale the output projection ahead of time during inference for higher generation throughput.
    if (
        self.per_expert_scale is not None
        and self.config.model_call_mode == "inference"
        and self.config.fuse_expert_scales
    ):
      self.wo.value = self.wo.value * self.per_expert_scale.value[:, None, None]

  def _maybe_shard_with_logical(self, inputs, logical_name):
    return maybe_shard_with_logical(
        inputs,
        logical_name,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def _logical_to_mesh_axes(self, logical_name):
    logical_rules = get_logical_axis_rules()
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def _maybe_shard_with_pspec(self, inputs, pspec: jax.sharding.PartitionSpec | None):
    return maybe_shard_with_pspec(
        inputs,
        pspec,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def _maybe_shard_moe_dispatch(self, inputs, logical_axis, peel_expert):
    """Shard a MoE dispatch/MLP activation. When `peel_expert` is set, drop the 'expert'
    mesh axis from the batch dim (index 1) so the GEMM stays expert-parallel (AllToAll)
    instead of double-mapping E and B onto 'expert'. Each logical dim is resolved
    independently so the shared 'expert' axis is not deduped off the expert dim before
    the peel."""
    if not peel_expert:
      return self._maybe_shard_with_logical(inputs, logical_axis)
    spec = [None if name is None else self._logical_to_mesh_axes((name,))[0] for name in logical_axis]
    pspec = remove_expert_from_partition_spec(jax.sharding.PartitionSpec(*spec), dims_to_peel=(1,))
    return self._maybe_shard_with_pspec(inputs, pspec)

  def get_expert_parallelism_size(self):
    # When expert parallelism has more than one physical axes, take product of their shapes
    if isinstance(self._expert_parallelism_name, tuple):
      return math.prod(self.mesh.shape.get(name, 1) for name in self._expert_parallelism_name)
    return self.mesh.shape.get(self._expert_parallelism_name, 1)

  def get_tensor_parallelism_size(self):
    if isinstance(self._tensor_parallelism_name, tuple):
      size = 1
      for axis in self._tensor_parallelism_name:
        size *= self.mesh.shape.get(axis, 1)
      return size
    return self.mesh.shape.get(self._tensor_parallelism_name, 1)

  def get_tensor_transpose_parallelism_size(self):
    return self.mesh.shape.get("tensor_transpose", 1)

  def get_context_autoregressive_parallelism_size(self):
    return self.mesh.shape.get("context_autoregressive", 1)

  def should_update_load_balance(self):
    """Determines if loss-free load balancing updates should be applied.

    The bias update logic is only applicable to Top-K router.
    Hash router does not use routed bias.
    """
    return self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and not self.is_hash_routing

  def get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None, saved_indices=None):
    """get topk.

    ``saved_indices`` (moe_save_sort_indices): the FORWARD's top_k_indices, saved through the
    hand-written backward's residuals. The index SEARCH (top_k / group masking / randint) is
    skipped and the weights are RE-DERIVED from the live logits with the same take_along_axis
    each routing mode uses -- identical values AND an identical (differentiable) gate-gradient
    path, so loss and grads are bit-exact vs recomputing. (Saving the weights themselves as
    constants would zero the gate/router gradient.)
    """
    # shape of top_k_weights & top_k_indices:
    # (batch, sequence, num_experts_per_tok).
    if self.config.use_random_routing:
      if saved_indices is not None:
        # random_routing's weights are exactly take_along_axis(gate_logits, indices); mirror its
        # early return (no scaling tail).
        return jnp.take_along_axis(gate_logits, saved_indices, axis=-1), saved_indices
      if self.config.moe_routing_key_as_input:
        # Constant-seed key, derived in-scope (pure jax, no rng state): byte-identical when the MoE
        # is re-traced (e.g. a hand-written layer backward recomputing routing). Routing is frozen
        # across steps by construction.
        rng = jax.random.key(self.config.moe_random_routing_seed)
      else:
        if rngs is None:
          raise ValueError("The random key cannot be None for random routing.")
        # Reuse the 'params' RNG stream to ensure random routing
        rng = rngs.params() if hasattr(rngs, "params") and callable(getattr(rngs, "params")) else rngs
      top_k_weights, top_k_indices = random_routing(rng, gate_logits, self.num_experts_per_tok)
      return top_k_weights, top_k_indices

    if saved_indices is not None:
      top_k_indices = saved_indices
      if self.is_hash_routing or self.config.model_name.startswith(("deepseek3", "deepseek4")):
        # hash routing and deepseek_routing both weight via take_along_axis(pre_bias_logits, idx).
        top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
      elif self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
        router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
        top_k_weights = jnp.take_along_axis(router_probs, top_k_indices, axis=-1).astype(self.dtype)
      else:
        # jax.lax.top_k's values are gate_logits at the top-k indices, in index order.
        top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
    elif self.is_hash_routing:
      if input_ids is None:
        raise ValueError("input_ids cannot be None when is_hash_routing is True")
      # Access the static routing table
      tid2eid_int = self.tid2eid.value
      # Cast the float32 array to int32 (JAX automatically assigns 0.0 gradients to integer casts)
      tid2eid_int = tid2eid_int.astype(jnp.int32)
      # Cast input_ids to int32 to safely index the hash routing table
      top_k_indices = tid2eid_int[input_ids.astype(jnp.int32)]
      top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    # NOTE: deepseek2 has a different pattern
    elif self.config.model_name.startswith(("deepseek3", "deepseek4")):
      top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
    elif self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
      _, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
      top_k_weights = jnp.take_along_axis(router_probs, top_k_indices, axis=-1).astype(self.dtype)
    else:
      top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block in (ctypes.DecoderBlockType.DEEPSEEK, ctypes.DecoderBlockType.DEEPSEEK4):
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    else:
      if self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4):
        top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

      # Normalization of router weights (e.g. used by Qwen3, Gemma4).
      if self.config.norm_topk_prob:
        top_k_weights /= top_k_weights.sum(axis=-1, keepdims=True)

    return top_k_weights, top_k_indices

  def deepseek_scale_weights(self, weights):
    """Scales weights according to DeepSeek's v3 reference implementation."""
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594.
    if self.config.routed_score_func in ("sigmoid", "sqrtsoftplus"):
      weights /= weights.sum(-1, keepdims=True) + 1e-20
    weights *= self.config.routed_scaling_factor
    return weights

  def expert_group_mask(self, gate_logits: jax.Array) -> jax.Array:
    """Returns a mask that selects only the top-k groups of experts.

    Groups of experts are selected based on the sum of the top-2 expert scores
    for each group.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.

    Returns:
      Array of shape `(batch, seq, num_experts)` that is 1 for experts in the
      top-k groups and 0 elsewhere.
    """
    # Find top groups based on each group's top-2 expert scores, where
    # `scores_grouped.shape =
    # (batch * seq, n_routing_groups, experts_per_group)`.
    scores_grouped = jnp.reshape(
        gate_logits,
        gate_logits.shape[:-1] + (self.config.n_routing_groups, -1),
    )
    top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
    group_scores = jnp.sum(jnp.astype(top2_in_group_vals, jnp.float32), axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, k=self.config.topk_routing_group)

    # Mask selected groups so that only those experts are considered.
    group_mask = jax.nn.one_hot(group_idx, num_classes=self.config.n_routing_groups, dtype=jnp.float32)
    group_mask = jnp.sum(group_mask, axis=-2)

    # Apply masks and get top-k indices.
    score_mask_expanded = jnp.broadcast_to(
        group_mask[..., None],
        group_mask.shape + (self.num_experts // self.config.n_routing_groups,),
    )
    return jnp.reshape(
        score_mask_expanded,
        score_mask_expanded.shape[:-2] + (self.num_experts,),
    )

  def deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """DeepSeek routing logit.

    If the configuration does not specify routing groups (`n_routing_groups` is
    -1), we use a standard top-k routing mechanism. Otherwise, we force all
    selected experts to be from the a subset of the highest rated expert groups.

    The selection process uses post_bias logits, while the return weights use
    pre_bias logits.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.
      pre_bias_logits: Array of shape `(batch, seq,num_experts)`.

    Returns:
      - top_k_weights: `(batch, seq, num_experts_per_tok)` array of weight values for
        each selected expert.
      - top_k_indices: `(batch, seq, num_experts_per_tok)` array of indices
        identifying the selected experts for each token.
    """
    expert_mask = 1 if self.config.n_routing_groups == -1 else self.expert_group_mask(gate_logits)
    _, top_k_indices = jax.lax.top_k(
        jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
        k=self.num_experts_per_tok,
    )
    top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def apply_ffn_activation(self, layer_w0, layer_w1):
    """Applies FFN activation function."""
    with jax.named_scope("ffn_act"):
      if self.config.decoder_block == ctypes.DecoderBlockType.GPT_OSS:
        layer_w0 = jnp.clip(layer_w0, min=None, max=self.config.mlp_activations_limit)
        layer_w1 = jnp.clip(
            layer_w1,
            min=-self.config.mlp_activations_limit,
            max=self.config.mlp_activations_limit,
        )
        layer_act = self.activation_fn(layer_w0 * 1.702)
        glu = jnp.multiply(layer_w0, layer_act)
        intermediate_layer = jnp.multiply(glu, (layer_w1 + 1))
      elif (
          self.config.decoder_block in (ctypes.DecoderBlockType.DEEPSEEK, ctypes.DecoderBlockType.DEEPSEEK4)
          and self.config.mlp_activations_limit > 0.0
      ):
        # DeepSeek V4 uses bounds to clip the SwiGLU activations
        layer_w0 = jnp.clip(layer_w0, min=None, max=self.config.mlp_activations_limit)
        layer_w1 = jnp.clip(
            layer_w1,
            min=-self.config.mlp_activations_limit,
            max=self.config.mlp_activations_limit,
        )
        layer_act = self.activation_fn(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
      else:
        layer_act = self.activation_fn(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)
      return intermediate_layer.astype(self.dtype)

  def permute(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      use_custom_sort_vjp=True,
      rngs=None,
      roll_to_expert_id=None,
      input_ids=None,
      dispatch_x_is_local=False,
      saved_sort=None,
      sort_save_cell=None,
  ):
    """Permute tokens to group by expert to fit gmm call.

    `dispatch_x_is_local` (decouple_dispatch_chunks, rung 9): when True, `inputs` is the PRE-AG
    LOCAL x (routing tensors gate_logits/pre_bias_logits are still GLOBAL) and the ragged
    dispatch is done by `chunked_ring_dispatch`, which chunks the token AG internally. The GLOBAL
    token count then comes from gate_logits, not from the (local) inputs.

    moe_save_sort_indices: `sort_save_cell` (a dict; forward CAPTURE) makes this compute the
    ring-sort's int index bundle in-line, feed it to ring_ragged_sort(precomputed_sort=...)
    (bit-identical output), and store `(top_k_indices, token_indices_sorted, group_sizes,
    revert_indices)` in the cell. `saved_sort` (backward CONSUME) is that bundle: the top-k
    search and the sort's argsorts/one-hot are skipped, weights are re-derived from the live
    logits (differentiable; see get_topk). Only valid on the ring ragged-sort path.
    """
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    inputs_2d = jnp.reshape(inputs, (inputs_shape[0] * inputs_shape[1], inputs_shape[2]))
    # Token count for routing/buffer sizing: GLOBAL. Normally inputs is the all-gathered global x
    # so this equals inputs_shape[0]*inputs_shape[1]; with chunked dispatch inputs is LOCAL, so
    # take the global count from the (always-global) gate_logits instead.
    if dispatch_x_is_local:
      bsz_times_seq_len = gate_logits.shape[0] * gate_logits.shape[1]
    else:
      bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    weights, selected_experts = self.get_topk(
        gate_logits, pre_bias_logits, rngs, input_ids, saved_indices=None if saved_sort is None else saved_sort[0]
    )
    lb_loss = None
    if self.config.load_balance_loss_weight > 0.0 and not self.is_hash_routing:
      softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
      lb_loss = self.load_balance_loss(selected_experts, softmax_probs)

    if self.should_update_load_balance():
      bias_updates = calculate_load_balance_updates(
          selected_experts,
          self.config.num_experts,
          self.config.routed_bias_update_rate,
      )
    else:
      bias_updates = None

    if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
      # weights will be of shape (batch_size, seq_len, num_experts_per_tok)
      router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))  # weights are top_k_weights here
      # Squeeze router_scores to (batch_size * seq_len, num_experts_per_tok)
      inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

    num_expert_parallelism = self.get_expert_parallelism_size()
    # The ragged-kernel path inside permute()/unpermute() is only correct for
    # the ring-of-experts strategy: each shard's output is masked to its own
    # [start, end) range within a globally-sorted layout. When ring of experts
    # is disabled, the buffer must instead carry all tokens for the subsequent
    # ragged-all-to-all, so we keep the standard argsort + sort path here and
    # let local_permute()/local_unpermute apply the ragged kernels on the
    # local prefix of valid rows.
    use_ragged_in_permute = self.config.use_ragged_sort and self.config.use_ring_of_experts
    buffer_size = None
    if use_ragged_in_permute:
      topk_indices_2d = jnp.reshape(selected_experts, (bsz_times_seq_len, selected_experts.shape[2]))
      # roll_to_expert_id is not directly used in the kernel, ep axis id is directly called
      if self.config.ragged_buffer_factor > 0.0:
        balanced_size = (bsz_times_seq_len // num_expert_parallelism) * self.num_experts_per_tok
        buffer_size = self.get_ragged_buffer_size(
            balanced_size,
            num_expert_parallelism,
            self.config.num_experts,
            self.num_experts_per_tok,
            self.config.ragged_buffer_factor,
        )
      else:
        buffer_size = None

      if dispatch_x_is_local:
        if saved_sort is not None or sort_save_cell is not None:
          raise ValueError(
              "moe_save_sort_indices is not supported with the chunked dispatch "
              "(decouple_dispatch_chunks>1): it computes its sort indices internally."
          )
        # inputs_2d is the PRE-AG LOCAL x; chunk the token AG + ragged-sort (rung 9). buffer_size
        # is None here (gated to ragged_buffer_factor<=0), so the full-buffer path is used.
        sorted_inputs, group_size, sorted_selected_experts = chunked_ring_dispatch(
            inputs_2d,
            topk_indices_2d,
            self.config.num_experts,
            self.num_experts_per_tok,
            self._expert_parallelism_name,
            num_expert_parallelism,
            self.config.decouple_dispatch_chunks,
            enforce_gather_fallback=self.config.ragged_gather_fallback,
            enforce_gather_reduce_fallback=self.config.ragged_gather_reduce_fallback,
            gather_flops_override=self.config.ragged_gather_cost_estimate_flops,
            gather_reduce_flops_override=self.config.ragged_gather_reduce_cost_estimate_flops,
            gather_bytes_accessed_override=self.config.ragged_gather_cost_estimate_bytes_accessed,
            gather_reduce_bytes_accessed_override=self.config.ragged_gather_reduce_cost_estimate_bytes_accessed,
        )
      else:
        precomputed_sort = None
        if saved_sort is not None:
          # BACKWARD CONSUME: the saved int bundle replaces the argsorts + one-hot group-size sum.
          precomputed_sort = (saved_sort[1], saved_sort[2], saved_sort[3])
        elif sort_save_cell is not None:
          # FORWARD CAPTURE: compute the bundle in-line (bit-identical to the in-kernel
          # computation) so it can be threaded out and saved as a residual.
          precomputed_sort = compute_ring_sort_indices(
              topk_indices_2d, self.config.num_experts, self.num_experts_per_tok
          )
          sort_save_cell["sort_bundle"] = (selected_experts,) + tuple(precomputed_sort)
        # moe_fp8_dispatch_wire: quantize the dispatch tokens to e4m3 (global scale) so the ring
        # dispatch gather moves HALF the wire bytes (ragged_gather is dtype-agnostic, packing 2->4),
        # then dequant back to bf16 right after the sort (the GMM re-quantizes in-kernel as today).
        # Wire-only: the k-way combine reduce stays f32/bf16. amax here is a full reduction (movable
        # to the RMSNorm epilogue later to hide it). Gated, default off.
        _fp8_wire = getattr(self.config, "moe_fp8_dispatch_wire", False) and isinstance(
            self._expert_parallelism_name, str
        )
        _dispatch_in = inputs_2d
        if _fp8_wire:
          _wire_scale = (jnp.max(jnp.abs(inputs_2d)).astype(jnp.float32) / 448.0 + 1e-20)
          _dispatch_in = (inputs_2d / _wire_scale.astype(inputs_2d.dtype)).astype(jnp.float8_e4m3fn)
        sorted_inputs, group_size, sorted_selected_experts = ring_ragged_sort(
            _dispatch_in,
            topk_indices_2d,
            self.config.num_experts,
            self.num_experts_per_tok,
            self._expert_parallelism_name,
            num_expert_parallelism,
            buffer_size=buffer_size,
            enforce_gather_fallback=self.config.ragged_gather_fallback,
            enforce_gather_reduce_fallback=self.config.ragged_gather_reduce_fallback,
            gather_flops_override=self.config.ragged_gather_cost_estimate_flops,
            gather_reduce_flops_override=self.config.ragged_gather_reduce_cost_estimate_flops,
            gather_bytes_accessed_override=self.config.ragged_gather_cost_estimate_bytes_accessed,
            gather_reduce_bytes_accessed_override=self.config.ragged_gather_reduce_cost_estimate_bytes_accessed,
            precomputed_sort=precomputed_sort,
            use_single_sparsecore=self.config.ragged_sort_use_single_sparsecore,
        )
        if _fp8_wire:
          # dequant the e4m3-dispatched tokens back to bf16 (the GMM quantizes in-kernel as today)
          sorted_inputs = sorted_inputs.astype(inputs_2d.dtype) * _wire_scale.astype(inputs_2d.dtype)
    else:
      if saved_sort is not None or sort_save_cell is not None:
        raise ValueError("moe_save_sort_indices requires the ring ragged-sort path (use_ragged_sort + ring of experts).")
      flatten_selected_experts = jnp.ravel(selected_experts)

      if roll_to_expert_id is not None:
        flatten_selected_experts = (flatten_selected_experts - roll_to_expert_id) % self.num_experts
      sorted_selected_experts = jnp.argsort(flatten_selected_experts)
      # sort inputs for number of selected experts
      replicated_inputs_2d = jnp.repeat(inputs_2d, self.num_experts_per_tok, axis=0)
      sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp).astype(
          self.dtype
      )
      group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)

    num_tokens = bsz_times_seq_len * self.num_experts_per_tok
    use_truncated_buffer = use_ragged_in_permute and buffer_size is not None and buffer_size < num_tokens

    if use_truncated_buffer:
      local_num_experts = self.config.num_experts // num_expert_parallelism
      shard_idx = jax.lax.axis_index(self._expert_parallelism_name) if num_expert_parallelism > 1 else 0
      experts_start = shard_idx * local_num_experts
      local_group_size = jax.lax.dynamic_slice_in_dim(
          group_size,
          experts_start,
          local_num_experts,
          axis=0,
      )
      # Clamp local_group_size to buffer_size to ensure we don't exceed buffer
      # capacity by leveraging the helper _truncate_matrix.
      local_group_size = _truncate_matrix(local_group_size[:, None], buffer_size)[:, 0]
      expert_indices = jnp.arange(local_num_experts)
      sorted_experts = jnp.repeat(
          expert_indices,
          repeats=local_group_size,
          total_repeat_length=buffer_size,
      )
    else:
      local_group_size = None
      expert_indices = jnp.arange(self.num_experts)
      sorted_experts = jnp.repeat(
          expert_indices,
          repeats=group_size,
          total_repeat_length=math.prod(selected_experts.shape),
      )

    return (
        sorted_inputs,
        sorted_selected_experts,
        weights,
        group_size,
        sorted_experts,
        lb_loss,
        bias_updates,
        local_group_size,
    )

  def unpermute(
      self,
      intermediate,
      sorted_selected_experts,
      weights,
      batch_size,
      sequence_length,
      use_custom_sort_vjp=True,
      group_sizes=None,
  ):
    """Unpermute tokens to original order and combine weights."""

    if self.config.use_ragged_sort and self.config.use_ring_of_experts:
      local_num_experts = self.config.num_experts // self.get_expert_parallelism_size()
      # Build the flat routing weights in the same layout as
      # topk_argsort_revert_indices (i.e. the flat token×topk order before
      # sorting by expert). `weights` has shape (batch, seq, topk); flatten it.
      flat_weights = jnp.ravel(weights).astype(jnp.float32)
      output = ring_ragged_unsort(
          intermediate,
          group_sizes,
          sorted_selected_experts,
          self.num_experts_per_tok,
          local_num_experts,
          self._expert_parallelism_name,
          topk_weights=flat_weights,
          enforce_gather_fallback=self.config.ragged_gather_fallback,
          enforce_gather_reduce_fallback=self.config.ragged_gather_reduce_fallback,
          gather_flops_override=self.config.ragged_gather_cost_estimate_flops,
          gather_reduce_flops_override=self.config.ragged_gather_reduce_cost_estimate_flops,
          gather_bytes_accessed_override=self.config.ragged_gather_cost_estimate_bytes_accessed,
          gather_reduce_bytes_accessed_override=self.config.ragged_gather_reduce_cost_estimate_bytes_accessed,
          use_single_sparsecore=self.config.ragged_sort_use_single_sparsecore,
      )
    else:
      unsort_intermediate = _sort_activations(
          intermediate,
          jnp.argsort(sorted_selected_experts),
          use_custom_sort_vjp,
      )
      reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
      reshaped_intermediate = jnp.reshape(
          unsort_intermediate,
          (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
      )
      with jax.named_scope("weight_sum"):
        matmul_precision = jax.lax.Precision(self.config.matmul_precision)
        if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
          # For Llama4, combine using weights of 1 for selected experts
          reshaped_weights = jnp.ones_like(reshaped_weights)
        if self.config.float32_weight_sum:
          reshaped_intermediate = reshaped_intermediate.astype(jnp.float32)
          reshaped_weights = reshaped_weights.astype(jnp.float32)
        output = jnp.einsum(
            "BKE,BK -> BE",
            reshaped_intermediate,
            reshaped_weights,
            precision=matmul_precision,
        )
    return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

  @staticmethod
  def _maybe_truncate_local_group_size(
      all_shard_local_sizes: jax.Array,
      buffer_size: int,
      ragged_buffer_factor: float,
  ) -> jax.Array:
    """Optionally truncates the local group sizes if ragged_buffer_factor > 0.0.

    When ragged_buffer_factor > 0, the receiver buffer has a fixed capacity
    (buffer_size). Due to routing imbalance, some shards might receive more
    tokens
    than this capacity. We use a prefix sum to deterministically truncate the
    received tokens on all shards, ensuring we don't write out of bounds.
    """
    if ragged_buffer_factor > 0.0:
      flat_sizes = all_shard_local_sizes.reshape(-1)
      cumsum = jnp.cumsum(flat_sizes)
      clamped_cumsum = jnp.minimum(cumsum, buffer_size)
      clamped_cumsum_extended = jnp.concatenate([jnp.zeros((1,), dtype=clamped_cumsum.dtype), clamped_cumsum])
      truncated_flat_sizes = jnp.diff(clamped_cumsum_extended)
      truncated_all_shard_local_sizes = truncated_flat_sizes.reshape(all_shard_local_sizes.shape)
      return jnp.sum(truncated_all_shard_local_sizes, axis=0)
    else:
      return jnp.sum(all_shard_local_sizes, axis=0)

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
      use_custom_sort_vjp=True,
      use_ragged_sort=False,
      ragged_buffer_factor=-1.0,
      use_single_sparsecore=False,
  ):
    """Permutes tokens locally within an expert shard.

    This function prepares the input tokens for processing by the experts
    located
    on the current shard. It groups the tokens by their assigned local expert
    index (0 to local_expert_size - 1).

    Args:
      inputs: The input data (tokens) assigned to the experts on this shard.
        Shape `[tokens, emb_dim]`.
      global_group_sizes: The count of tokens assignments for each global expert
        across all the batch shards. Shape `[num_batch_shards, num_experts].
      local_expert_size: The number of experts handled by the current shard.
      shard_index: The index of the current expert shard (0 to
        num_expert_parallelism - 1).
      is_offset: If True, assumes `inputs` are pre-sorted by global expert ID
        and selects the slice relevant to this shard's assigned experts. If
        False, assumes that `inputs` corresponding to the shard's experts start
        from the beginning of the tensor but need to be permuted by expert ID.
      global_sorted_experts: Global expert IDs for the `inputs` used when
        `is_offset` is True. Shape `[total_tokens_for_this_shard]`.
      use_custom_sort_vjp: Whether to use the explicit custom-VJP gather/scatter
        for the standard sort path. Ignored when `use_ragged_sort=True`.
      use_ragged_sort: When True, use the Pallas ragged-gather kernel
        (`a2a_ragged_sort`) to sort only the valid prefix of `inputs`. The
        ragged buffer can be much larger than the actually-routed token count,
        so this avoids touching the padded tail in both forward and backward.

    Returns:
      A tuple containing:
        sorted_inputs: Input data permuted local expert ID.
        sorted_indices: Indices used to permute the inputs.
        local_group_size: Number of tokens assigned to each local expert on this
          shard.
        sorted_experts_ids: expert ID corresponding to each token of the permuted
        inputs.
    """

    # Slice the count of local expert IDs in each batch shard.
    # all_shard_local_sizes.shape: [expert_shard, local_expert_size]
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        shard_index * local_expert_size,
        local_expert_size,
        axis=1,
    )
    local_sizes = all_shard_local_sizes.reshape(-1)

    # Total count of the local expert IDs is the sum of the counts across all
    # batch shards, since all batch shards will send their contributions to the
    # current expert shard.
    local_group_size = RoutedMoE._maybe_truncate_local_group_size(
        all_shard_local_sizes, inputs.shape[0], ragged_buffer_factor
    )

    # In this case, the data that needs to be processed by the local shard
    # does not start from row 0 but actually starts at
    # (jnp.concatenate((jnp.array([0]),
    #  jnp.cumsum(local_group_sizes[:-1]))[shard_id]).
    # This happens if batches (`inputs`) are replicated across expert shards and
    # pre-sorted by global Expert ID (via permute()).
    if is_offset:
      divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
      expert_indices = jnp.where(
          divided_assignments == shard_index,
          jnp.mod(global_sorted_experts, local_expert_size),
          local_expert_size,
      )

    # In this case the `input` data has been received from the batch shards and
    # needs to be reorganized in order of local Expert IDs.
    else:
      base_indices = jnp.mod(jnp.arange(local_sizes.shape[0]), local_expert_size)
      expert_indices = jnp.repeat(base_indices, local_sizes, total_repeat_length=inputs.shape[0])

    sorted_indices = jnp.argsort(expert_indices)
    if use_ragged_sort:
      # Only the first `valid_end` rows of `inputs` carry actual tokens for
      # this shard (`local_group_size.sum()`), the remainder is padding from
      # the worst-case ragged buffer. Restricting the gather to that prefix
      # makes both forward and backward proportional to the routed token count.
      valid_end = jnp.sum(local_group_size).astype(jnp.int32)
      sorted_inputs = a2a_ragged_sort(
          inputs,
          sorted_indices,
          valid_end,
          use_single_sparsecore=use_single_sparsecore,
      )
    else:
      sorted_inputs = _sort_activations(inputs, sorted_indices, use_custom_sort_vjp)
    sorted_experts_ids = expert_indices[sorted_indices]
    return (
        sorted_inputs,
        sorted_indices,
        local_group_size,
        sorted_experts_ids,
    )

  @staticmethod
  def get_all_to_all_params(
      all_shards_group_sizes,
      shard_id,
      num_expert_parallelism,
      is_batch_sharded=True,
      ragged_buffer_factor=-1.0,
      buffer_size=None,
      is_dispatch=True,
  ):
    """Generates input offsets, send sizes, output offsets, and receive sizes used for ragged_all_to_all."""

    class TransformStrategy(enum.Enum):
      INPUT_OFFSET = enum.auto()
      SEND_SIZE = enum.auto()
      OUTPUT_OFFSET = enum.auto()
      RECV_SIZE = enum.auto()

    def transform_array(input_array, shard_id, strategy, is_batch_sharded):
      """Transforms the input array based on the specified strategy."""
      # Prepares it for the usage with `ragged_all_to_all` API. The
      # transformation determines how data is sent and received between shards.
      if is_batch_sharded:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # Index of input array for the send
          local_array = input_array[shard_id]
          return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
        elif strategy == TransformStrategy.SEND_SIZE:
          # Size of input array for the send
          return input_array[shard_id]
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # Received index in the target output
          zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
          array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
          cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
          return cumulated_array[shard_id]
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the target output
          return input_array[:, shard_id]
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

      # If the batch is unsharded then we send the same data slice to all other
      # shards. We also assume each shard will have the local processed inputs
      # sorted to start from index 0. Finally, len(input_array.shape) == 1 since
      # there is only one batch shard.
      else:
        if strategy == TransformStrategy.INPUT_OFFSET:
          # The data on each shard always starts at 0.
          return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
        elif strategy == TransformStrategy.SEND_SIZE:
          # The send amount is always the amount of data the current expert
          # shard needs to process.
          return jnp.repeat(input_array[shard_id], num_expert_parallelism)
        elif strategy == TransformStrategy.OUTPUT_OFFSET:
          # The offset in each shard will just be the start of the group which
          # that shard is responsible for.
          output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
          return jnp.repeat(output_offset, num_expert_parallelism)
        # The amount that each shard receives from all other shards is
        # equivalent to the group sizes (aka input_array).
        elif strategy == TransformStrategy.RECV_SIZE:
          # Received size in the target output
          return input_array
        else:
          raise ValueError(f"Unknown transform array strategy: {strategy}")

    if ragged_buffer_factor > 0.0:
      assert buffer_size is not None
      truncated_all_shards_group_sizes = _truncate_matrix(all_shards_group_sizes, buffer_size)

      if is_dispatch:
        # For input_offsets, we use the untruncated group sizes because the
        # sender's buffer still contains all tokens (including dropped ones).
        input_offsets = transform_array(
            all_shards_group_sizes,
            shard_id,
            TransformStrategy.INPUT_OFFSET,
            is_batch_sharded,
        )
        # For send/recv sizes and output_offsets, we use truncated group sizes
        # to ensure we don't write out of bounds of the receiver's capacity.
        send_sizes = transform_array(
            truncated_all_shards_group_sizes,
            shard_id,
            TransformStrategy.SEND_SIZE,
            is_batch_sharded,
        )
        output_offsets = transform_array(
            truncated_all_shards_group_sizes,
            shard_id,
            TransformStrategy.OUTPUT_OFFSET,
            is_batch_sharded,
        )
        recv_sizes = transform_array(
            truncated_all_shards_group_sizes,
            shard_id,
            TransformStrategy.RECV_SIZE,
            is_batch_sharded,
        )
      else:
        transposed_all_shards = jnp.transpose(all_shards_group_sizes)
        transposed_truncated = jnp.transpose(truncated_all_shards_group_sizes)

        # In combine stage, the roles are reversed:
        # input_offsets/sizes and recv_sizes use truncated parameters because
        # the combine sender buffer (dispatch receiver buffer) is packed.
        input_offsets = transform_array(
            transposed_truncated,
            shard_id,
            TransformStrategy.INPUT_OFFSET,
            is_batch_sharded,
        )
        send_sizes = transform_array(
            transposed_truncated,
            shard_id,
            TransformStrategy.SEND_SIZE,
            is_batch_sharded,
        )
        # output_offsets use untruncated parameters because we write back
        # to their original untruncated positions.
        output_offsets = transform_array(
            transposed_all_shards,
            shard_id,
            TransformStrategy.OUTPUT_OFFSET,
            is_batch_sharded,
        )
        recv_sizes = transform_array(
            transposed_truncated,
            shard_id,
            TransformStrategy.RECV_SIZE,
            is_batch_sharded,
        )
    else:
      matrix = all_shards_group_sizes if is_dispatch else jnp.transpose(all_shards_group_sizes)
      input_offsets = transform_array(
          matrix,
          shard_id,
          TransformStrategy.INPUT_OFFSET,
          is_batch_sharded,
      )
      send_sizes = transform_array(
          matrix,
          shard_id,
          TransformStrategy.SEND_SIZE,
          is_batch_sharded,
      )
      output_offsets = transform_array(
          matrix,
          shard_id,
          TransformStrategy.OUTPUT_OFFSET,
          is_batch_sharded,
      )
      recv_sizes = transform_array(
          matrix,
          shard_id,
          TransformStrategy.RECV_SIZE,
          is_batch_sharded,
      )

    return input_offsets, send_sizes, output_offsets, recv_sizes

  def transform_bias(self, experts_index, *biases):
    """Selects bias values for a variable number of bias tensors based on chosen experts."""
    return tuple(bias[experts_index] for bias in biases)

  @staticmethod
  def get_ragged_buffer_size(local_batch, ep_degree, global_experts, top_k, ragged_buffer_factor):
    """Calculates the token batch size of the ragged buffer.
    When explicitly setting ragged_buffer_factor>0, this is balanced_size * ragged_buffer_factor, which can drop tokens.
    Otherwise this will be worst case size to ensure no dropping.

    Inputs:
      local_batch: local token batch (batch*seq blown up by top_k) shard on this device (e.g. inside shard_map)
      ep_degree: degree of expert parallelism, generally equal to ici_expert_parallelism
      global_experts: unsharded expert count, e.g. 256 for deepseek
      top_k: aka num_experts_per_tok, 8 for deepseek.
      ragged_buffer_factor: When set > 0, the buffer is balanced_size * ragged_buffer_factor.
        The value 1.0 will be dropless only in the perfectly balanced case, else tokens will be dropped.
    Outputs:
      The ragged buffer's token batch size.
    """
    balanced_size = local_batch
    if ragged_buffer_factor > 0.0:
      # This will drop tokens if the true distribution exceeds this buffer.
      return int(balanced_size * ragged_buffer_factor)
    else:
      # Worst case
      # Either determined by degree of EP, or can be less when num_local_exp is smaller than top_k:
      # Example: If we have 4 EP shards, top_k=8, and experts=256 (deepseek), then worst case is
      # all tokens in our EP replica get routed to a single shard, e.g. rank 0 - thus is |EP|=4x larger than perfectly
      # balanced. However if we use EP=128, then there are only 256/128 = 2 local experts, and thus at most in an EP
      # replica group only the 2 experts of top_k=8 can be chosen, so at most 1/4 of all tokens goes to the most
      # popular shard. Thus the imbalance factor goes like |EP|/(top_k/local_exp) = 128/4 = 32.
      # In general for local_experts < top_k (e.g. |EP|>32), the balance will go as
      # EP * local_experts / top_k = EP * (global_exp/EP) / top_k = global_exp / top_k.
      # This is constant as a function of the model - e.g. for deepseek the imbalance is never worse than
      # 256 exp / 8 top_k = 32. In practice the imbalance should be much less and potentially can use
      # ragged_buffer_factor set to >1  e.g. 3.0, and likely have no dropping (not guaranteed)
      worst_case_factor = min(ep_degree, global_experts / top_k)
      return int(balanced_size * worst_case_factor)

  def sparse_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
      input_ids=None,
      use_chunked_combine=True,
      use_chunked_dispatch=True,
      return_combine_token=False,
      save_routing=False,
      saved_routing=None,
      bwd_direct_token_ag=False,
  ):
    """Perform sparse matrix multiplication of inputs and Experts.

    `use_chunked_combine` (static bool) gates the decouple_combine_rs_chunks combine path; the
    moe_handwritten_bwd recompute passes False so the backward differentiates the un-chunked
    combine (forward-only chunking in rung 6).

    `return_combine_token` (static bool, moe_shared_after_combine): when True, a 4th output is
    returned -- a [1, 1] SCHEDULING TOKEN sliced from the first chunk's pre-RS combined output
    of the decoupled chunked combine, or None when the emitting path is inactive. Fencing a
    consumer (the shared-expert MLP input) on the token delays it until the combine phase has
    begun WITHOUT depending on any reduce-scatter.

    moe_save_sort_indices: `save_routing` (static bool, forward) appends one more output -- a
    tuple over num_moe_token_chunks of `(top_k_indices, token_indices_sorted, group_sizes, revert)`
    int32 bundles (the ring ragged-sort's index computation, captured per chunk). All four are
    REPLICATED over the expert axis (computed from the EP-all-gathered logits), so their
    out_specs are the input batch spec MINUS the expert axis; group_sizes gets a leading
    size-1 axis to carry its per-(data/fsdp)-shard values across the boundary. `saved_routing`
    (backward recompute) feeds that bundle back in with the SAME specs: the recompute then
    skips the top-k search and the sort's argsorts + one-hot group-size sum (weights are
    re-derived differentiably; see get_topk/permute).
    """
    # Static gate for emitting the combine scheduling token: exactly the conditions under
    # which _moe_body takes the decoupled chunked-combine branch (plus num_moe_token_chunks <= 1:
    # the chunked-body loop calls _moe_body once per sequence chunk and does not emit).
    emit_combine_token = (
        return_combine_token
        and self.config.use_ring_of_experts
        and self.config.decouple_combine_rs_chunks > 1
        and use_chunked_combine
        and isinstance(self._expert_parallelism_name, str)
        and self.config.num_moe_token_chunks <= 1
    )

    def jax_ragged_dot_gmm(inputs, kernel, tiling, group_sizes, expert_assignments, padding_amount, group_offset=0):
      """Execute jax.lax.ragged_dot, with potential quantization"""
      m, k, n = inputs.shape[0], inputs.shape[1], kernel.shape[2]
      # Clamps the tile size using the minimum
      tiling = (
          min(tiling[0], m),
          min(tiling[1], k),
          min(tiling[2], n),
      )
      rhs_inputs = kernel
      if isinstance(kernel, aqt.QTensor):
        if kernel.bias or kernel.sparsity_mask or len(kernel.scale) > 1:
          raise ValueError("Unsupported usecase for ragged_dot with quantized kernel.")
        rhs_inputs = kernel.qvalue
      # Ring-of-experts EP>1 fallback (CPU/GPU reference): the kernel holds only this shard's
      # LOCAL experts while group_sizes covers all GLOBAL experts (megablox/tokamax handle this
      # via group_offset; jax.lax.ragged_dot has no such parameter and previously raised).
      # inputs is the GLOBAL expert-sorted buffer, so roll the shard's rows (starting at the
      # group_offset expert's cumulative offset) to the front, run ragged_dot with the LOCAL
      # group sizes, and roll back. Rows outside the shard's valid range are don't-care
      # (masked by the downstream combine), matching the TPU kernels' unwritten rows.
      unshift = None
      if group_sizes.shape[0] != rhs_inputs.shape[0]:
        if isinstance(kernel, aqt.QTensor):
          raise ValueError("group_offset ragged_dot fallback does not support quantized kernels.")
        offsets = jnp.cumulative_sum(group_sizes.astype(jnp.int32), include_initial=True)
        unshift = offsets[group_offset]
        group_sizes = jax.lax.dynamic_slice_in_dim(group_sizes, group_offset, rhs_inputs.shape[0], axis=0)
        inputs = jnp.roll(inputs, -unshift, axis=0)
      if self.config.quantization and self.config.use_qwix_quantization:
        # Use full contraction for QWIX quantization to allow quantization
        # fusion (max reduce over contracting dimension).
        tiling = (tiling[0], k, tiling[2])

      is_tpu = self.mesh.devices.flat[0] == "tpu"
      # TPU needs random mosaic_fusion_group; GPU/CPU needs deterministic ID for autotuner sync
      mosaic_group_id = f"{random.randint(0, 1000000000)}" if is_tpu else "0"
      with set_xla_metadata(
          ragged_dot_tiling=",".join([str(t) for t in tiling]),
          mosaic_fusion_group=mosaic_group_id,
      ):
        output = jax.lax.ragged_dot(
            lhs=inputs,
            rhs=rhs_inputs,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
        )
      if isinstance(kernel, aqt.QTensor):
        # Multiply outputs by the kernely scale
        scales = jnp.take(kernel.scale[0].squeeze(), indices=expert_assignments, axis=0)
        if padding_amount > 0:
          scales = jax.lax.pad(
              scales,
              jnp.array(0.0, dtype=scales.dtype),
              [(0, padding_amount, 0), (0, 0, 0)],
          )
        output *= scales
      if unshift is not None:
        output = jnp.roll(output, unshift, axis=0)
      return output

    def get_tokamax_group_sizes(group_sizes, inputs, _kernel):
      if self.config.quantization and self.config.use_qwix_quantization:
        return group_sizes
      elif self.config.attention == "vllm_rpa":
        return group_sizes
      else:
        num_groups = group_sizes.shape[0]
        return tokamax.RaggedDotGroupSizes(
            group_sizes,
            (inputs.shape[0] // num_groups,) * num_groups,
        )

    def get_quantization_dtypes():
      lhs_quantize_dtype, rhs_quantize_dtype = None, None
      if self.quant is not None:
        quant_dg = self.quant.quant_dg
        lhs_quantize_dtype = quant_dg.fwd.dg_quantizer.lhs.numerics.get_dtype()
        rhs_quantize_dtype = quant_dg.fwd.dg_quantizer.rhs.numerics.get_dtype()
      return lhs_quantize_dtype, rhs_quantize_dtype

    def gmm(inputs, kernel, tiling, group_sizes, expert_assignments, weight_gather_axes, group_offset):
      def extract_vma(tensor):
        # Parses the varying mesh axes from JAX's type string for a tensor inside shard_map.
        # jax.typeof(t) renders as e.g. 'f32[128,256]{V:(expert, fsdp)}'; this extracts
        # ('expert', 'fsdp'). Returns () if the tensor has no varying axes.
        type_str = str(jax.typeof(tensor))
        if "{V:" in type_str:
          start = type_str.index("{V:") + 3
          end = type_str.index("}", start)
          vma_content = type_str[start:end].strip("()")
          return tuple(sorted(a.strip() for a in vma_content.split(",")))
        return tuple()

      # moe_fp8_boundary_qag: the kernel may be a qwix QArray (e4m3 qvalue + scale) built at the gmm
      # call site. VMA is on the qvalue leaf, and we must NOT astype it to bf16 (that would dequant the
      # e4m3 wire); ops.gmm consumes the QArray (rhs=qvalue, rhs_scale=scale) and dequants in-kernel.
      _kernel_is_qarray = isinstance(kernel, qpl.QArray)
      lhs_vma_axes = extract_vma(inputs)
      rhs_vma_axes = extract_vma(kernel.qvalue if _kernel_is_qarray else kernel)
      if inputs.shape[0] != expert_assignments.shape[0]:
        raise ValueError("The number of input tokens must match the number of expert assignments!")

      tokamax_group_sizes = get_tokamax_group_sizes(group_sizes, inputs, kernel)
      orig_inputs_shape = inputs.shape  # save shape of inputs before potentially padding.
      inputs, padding_amount = max_utils.maybe_pad(inputs, self.config.wi_tile_fwd_batch_seq)
      inputs = inputs.astype(self.dtype)
      if not _kernel_is_qarray:
        kernel = kernel.astype(self.dtype)
      lhs_quantize_dtype, rhs_quantize_dtype = get_quantization_dtypes()

      # Interpret the megablox Pallas kernel only when the TARGET is NOT TPU (CPU or GPU,
      # e.g. equiv_chunk_test executing locally). During train_compile the local backend is
      # CPU (JAX_PLATFORMS=cpu) but self.mesh targets tpu7x -> compile natively; interpret
      # mode on a TPU target breaks check_vma and bloats HBM temporaries.
      megablox_interpret = self.mesh.devices.flat[0].platform != "tpu"

      # We support various implementations for gmm - tokamax gmm (v1, v2), older forked megablox, or jax.lax.ragged_dot
      # Determine whether we can use: tokamax gmm v1 (quantized)
      is_tokamax_v1_unquantized = (
          self.config.use_tokamax_gmm and not self.config.quantization and not self.config.use_gmm_v2
      )
      # Use custom vjp: tokamax gmm v1 (quantized), tokamax gmm v2 (quantized, unquantized), older forked megablox
      use_custom_vjp_gmm = self.config.use_tokamax_gmm or self.config.megablox

      if is_tokamax_v1_unquantized:
        # tokamax v1 (unquantized)
        output = tokamax.ragged_dot(
            lhs=inputs,
            rhs=kernel,
            group_sizes=tokamax_group_sizes,
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=self.dtype,
            implementation="mosaic",
            # `group_offset` is not yet supported
            group_offset=None,
        )
      elif use_custom_vjp_gmm:
        # tokamax gmm v1 (quantized), tokamax gmm v2 (quantized, unquantized), older forked megablox
        output = mblx.gmm(
            lhs=inputs,
            rhs=kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling,
            group_offset=group_offset,
            lhs_quantize_dtype=lhs_quantize_dtype,
            rhs_quantize_dtype=rhs_quantize_dtype,
            use_qwix_quantization=bool(self.config.quantization) and self.config.use_qwix_quantization,
            use_tokamax_backend=self.config.use_tokamax_gmm,
            weight_gather_axes=weight_gather_axes,
            lhs_vma_axes=lhs_vma_axes,
            rhs_vma_axes=rhs_vma_axes,
            use_gmm_v2=self.config.use_gmm_v2,
            interpret=megablox_interpret,
        )
      else:
        # jax.lax.ragged_dot
        output = jax_ragged_dot_gmm(
            inputs, kernel, tiling, group_sizes, expert_assignments, padding_amount, group_offset=group_offset
        )

      if padding_amount > 0:
        output = output[: orig_inputs_shape[0]]
      return output

    def is_batch_sharded_by_ep(input_activation):
      # The batch is sharded by expert, except during inference decoding (where batch size == 1).
      # In the decoding case, the expert axis is instead replicated along the tensor's batch dimension.
      return input_activation.shape[0] > 1

    def explicitly_weight_ag(shard_exp_on_fsdp):
      # moe_fp8_ring_weight_ag: also fire the in-GMM fp8 weight-AG (QAG) on the RING path
      # (not shard_exp_on_fsdp) so the FSDP embed-sharded weight is gathered as the e4m3 qvalue
      # (half the wire bytes) inside the GMM instead of the bf16 GSPMD boundary gather. Requires a
      # fixed (static) weight scale so only the qvalue rides the wire.
      _ring = getattr(self.config, "moe_fp8_ring_weight_ag", False)
      if shard_exp_on_fsdp or _ring:
        quantization_rule = qpl.get_current_rule("gmm")
        # Ring path (Option A) supports a DYNAMIC per-channel weight-AG -> allow any calibration.
        # shard_exp_on_fsdp still requires the fixed (static) scale of the stock QAG.
        if quantization_rule and (_ring or quantization_rule.weight_calibration_method.startswith("fixed")):
          return True
      return False

    def maybe_aqt_partition(w0_kernel, w0_pspec, w1_kernel, w1_pspec, wo_kernel, wo_pspec):
      if isinstance(w0_kernel, aqt.QTensor):
        w0_pspec = aqt.partition_spec(w0_pspec, (1,), w0_kernel.dtype, use_bias=False)
      if isinstance(w1_kernel, aqt.QTensor):
        w1_pspec = aqt.partition_spec(w1_pspec, (1,), w1_kernel.dtype, use_bias=False)
      if isinstance(wo_kernel, aqt.QTensor):
        wo_pspec = aqt.partition_spec(wo_pspec, (1,), wo_kernel.dtype, use_bias=False)
      return w0_pspec, w1_pspec, wo_pspec

    def get_routed_moe_shardings(is_batch_sharded_by_expert, has_input_ids):
      if is_batch_sharded_by_expert:
        batch_logical_axis = "activation_batch"
      else:
        batch_logical_axis = "decode_batch_moe"

      if self.get_tensor_transpose_parallelism_size() > 1:
        input_partition_pspec = self._logical_to_mesh_axes(
            (batch_logical_axis, "activation_norm_length", "activation_embed")
        )
        w0_bias_pspec = self._logical_to_mesh_axes(("exp", None))
        w1_bias_pspec = self._logical_to_mesh_axes(("exp", None))
        wo_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_embed"))
      else:
        input_partition_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
        w0_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_mlp"))
        w1_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_mlp"))
        wo_bias_pspec = self._logical_to_mesh_axes(("exp", "activation_embed"))

      gate_logits_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
      # NOTE: deepseek2 has a different pattern
      if self.config.model_name.startswith(("deepseek3", "deepseek4")):
        pre_bias_logits_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", None))
      else:
        # pre_bias_logits is None for non-deepseek3/4 models, including deepseek2
        pre_bias_logits_pspec = None

      if has_input_ids:
        decoder_tokens_pspec = self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length"))
      else:
        decoder_tokens_pspec = None

      # w0, w1, wo needs to be un sharded on fsdp / fsdp_transpose axis, so use
      # mlp_no_fsdp axis
      if self.config.shard_exp_on_fsdp:
        quantization_rule = qpl.get_current_rule("gmm")
        if quantization_rule and quantization_rule.weight_calibration_method.startswith("fixed"):
          # special sharding when using static scaling for weights in quantization with shard_exp_on_fsdp
          w0_pspec = self._logical_to_mesh_axes(self.wi_kernel_axes)
          w1_pspec = self._logical_to_mesh_axes(self.wi_kernel_axes)
          wo_pspec = self._logical_to_mesh_axes(self.wo_kernel_axes)
        else:
          # special sharding for dsv3 to remove overhead between gmm/AG
          w0_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", None, "mlp_no_fsdp"))
          w1_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", None, "mlp_no_fsdp"))
          wo_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
      elif self.config.use_2d_fsdp_sharding:
        w0_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
        w1_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
        wo_pspec = self._logical_to_mesh_axes(("embed_tensor_transpose", "mlp_no_fsdp", None))
      else:
        # These are the main shardings used by default - they use funky rules to AG over FSDP.
        w0_pspec = self._logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
        w1_pspec = self._logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
        wo_pspec = self._logical_to_mesh_axes(("exp", "mlp_no_fsdp", "embed_tensor_transpose"))
      return (
          batch_logical_axis,
          input_partition_pspec,
          gate_logits_pspec,
          pre_bias_logits_pspec,
          w0_pspec,
          w1_pspec,
          wo_pspec,
          w0_bias_pspec,
          w1_bias_pspec,
          wo_bias_pspec,
          decoder_tokens_pspec,
      )

    is_batch_sharded_by_expert = is_batch_sharded_by_ep(inputs)
    weight_gather = explicitly_weight_ag(self.config.shard_exp_on_fsdp)
    (
        batch_logical_axis,
        input_partition_pspec,
        gate_logits_pspec,
        pre_bias_logits_pspec,
        w0_pspec,
        w1_pspec,
        wo_pspec,
        w0_bias_pspec,
        w1_bias_pspec,
        wo_bias_pspec,
        decoder_tokens_pspec,
    ) = get_routed_moe_shardings(is_batch_sharded_by_expert, input_ids is not None)
    w0_pspec, w1_pspec, wo_pspec = maybe_aqt_partition(w0_kernel, w0_pspec, w1_kernel, w1_pspec, wo_kernel, wo_pspec)

    def route(x, logits, pre_bias_logits, rngs, input_ids=None, saved_sort=None, sort_save_cell=None):
      """Performs both across device and within device token routing/sorting"""
      num_ep = self.get_expert_parallelism_size()
      expert_shard_id = jax.lax.axis_index(self._expert_parallelism_name) if num_ep > 1 else 0

      local_sorted_indices = None
      all_shards_group_sizes = None
      reshaped_group_sizes = None

      if self.config.use_ring_of_experts:
        # The ring-of-experts strategy first duplicates the inputs to all
        # expert shards, and then routes within each shard.

        # DECOUPLED chunked dispatch (decouple_dispatch_chunks, rung 9): chunk the token AG over
        # the input-token axis so each chunk's all-gather hides under the previous chunk's
        # ragged-sort. Routing needs only the (small) logits gathered; x stays LOCAL and its AG
        # is chunked inside chunked_ring_dispatch. Gated to the plain ragged single-axis
        # full-buffer ring path (the chunked dispatch is the full-buffer variant). Off on the
        # moe_handwritten_bwd RECOMPUTE (use_chunked_dispatch=False) -> unchunked there.
        chunk_dispatch = (
            self.config.decouple_dispatch_chunks > 1
            and use_chunked_dispatch
            and self.config.use_ragged_sort
            and self._expert_parallelism_name == "expert"
            and self.config.ragged_buffer_factor <= 0
            and self.config.decoder_block != ctypes.DecoderBlockType.LLAMA4
        )
        if chunk_dispatch:
          # Gather ONLY the routing tensors; keep x LOCAL (its AG is chunked in permute).
          logits, pre_bias_logits = tuple(
              jax.lax.all_gather(z, axis_name=self._expert_parallelism_name, tiled=True)
              for z in (logits, pre_bias_logits)
          )
        elif bwd_direct_token_ag and isinstance(self._expert_parallelism_name, str):
          # moe_direct_token_ag (BACKWARD RECOMPUTE only, gated): all-gather the EP token/activation
          # `x` (bf16[tokens,embed], the big exposed gather) with the direct-to-owner TC Pallas
          # kernel instead of the XLA collective, so it rides the TensorCore ICI DMAs -- NOT the
          # SparseCore offload queue that serializes lax.all_gather behind the SC-resident weight
          # re-gather -- letting XLA overlap the two (different engines). Numerically ==
          # lax.all_gather (verified in isolation); its custom_vjp gives the same psum_scatter
          # transpose the collective would. The small routing logits stay on the plain collective.
          # Flag-off (bwd_direct_token_ag=False, and the whole forward) takes the tuple gather below
          # => byte-identical.
          x = _direct_all_gather(x, self.mesh, self._expert_parallelism_name, _DIRECT_TOKEN_AG_COLLECTIVE_ID)
          logits, pre_bias_logits = tuple(
              jax.lax.all_gather(z, axis_name=self._expert_parallelism_name, tiled=True)
              for z in (logits, pre_bias_logits)
          )
        elif self.config.moe_fwd_direct_token_ag and isinstance(self._expert_parallelism_name, str):
          # moe_fwd_direct_token_ag: run the FORWARD EP token dispatch all-gather of the big token tensor
          # `x` (bf16[tokens,embed]) with the direct-to-owner TensorCore Pallas kernel (_direct_all_gather)
          # instead of the XLA lax.all_gather -- moving it OFF the SparseCore offload queue (the 4.07s
          # binder) onto the TC ICI DMAs. The small routing logits stay on the plain collective. Its
          # custom_vjp gives the same psum_scatter transpose, so numerics == lax.all_gather. Symmetric to
          # moe_direct_token_ag (which does the BACKWARD recompute); this does the FORWARD dispatch.
          x = _direct_all_gather(x, self.mesh, self._expert_parallelism_name, _DIRECT_FWD_TOKEN_AG_COLLECTIVE_ID)
          logits, pre_bias_logits = tuple(
              jax.lax.all_gather(z, axis_name=self._expert_parallelism_name, tiled=True)
              for z in (logits, pre_bias_logits)
          )
        elif self.config.moe_wag_cotag_token and isinstance(self._expert_parallelism_name, str):
          # moe_wag_cotag_token: FORWARD-ONLY-tag the EP token dispatch all-gather into the weight-AG
          # scheduling group so XLA co-schedules it with the w0 FSDP weight all-gather (two SC-offload
          # collectives on independent ICI axes -> concurrent on the 2 SparseCores). The custom_vjp keeps
          # the BACKWARD transpose (a reduce-scatter) OUT of the group: a plain tagged all-gather would let
          # its RS inherit the tag, and a forward AG + its backward RS in one group closes a scheduling
          # CYCLE. Mirrors _make_cv_gather. Scheduling-only; numerics == lax.all_gather.
          ep_axis = self._expert_parallelism_name

          @jax.custom_vjp
          def _ep_g(z):  # PRIMAL: plain all-gather (what the backward recompute re-traces)
            return jax.lax.all_gather(z, axis_name=ep_axis, tiled=True)

          def _ep_g_fwd(z):  # FORWARD under diff: tagged all-gather
            with _scheduling_group(_WEIGHT_AG_SCHED_GROUP):
              out = jax.lax.all_gather(z, axis_name=ep_axis, tiled=True)
            return out, None  # no residual

          def _ep_g_bwd(_res, ct):  # transpose of a tiled all-gather (concat axis 0) = reduce-scatter, UNtagged
            return (jax.lax.psum_scatter(ct, axis_name=ep_axis, scatter_dimension=0, tiled=True),)

          _ep_g.defvjp(_ep_g_fwd, _ep_g_bwd)
          x, logits, pre_bias_logits = tuple(_ep_g(z) for z in (x, logits, pre_bias_logits))
        else:
          # Duplicate inputs to all expert shards.
          x, logits, pre_bias_logits = tuple(
              jax.lax.all_gather(z, axis_name=self._expert_parallelism_name, tiled=True)
              for z in (x, logits, pre_bias_logits)
          )

        # moe_x_sorted (option A): tag the PRE-duplication GATHERED tokens ([tokens_gathered, embed],
        # ~235MB/chunk at pdbs1) -- NOT the post-sort x_sorted, whose topk-8 row duplication makes it
        # 229GB across 61 layers (compile-OOM, measured 233.55G). With moe_x_sorted=device the
        # backward LOADS this tensor, killing the rematted EP dispatch all-gather; the SC ragged sort
        # still re-runs from it (the duplication IS the sort -- accepted). Inert under the default
        # moe_x_sorted=remat. In the chunk_dispatch branch x stays local (tag harmless there).
        x = adc.checkpoint_name(x, "moe_x_sorted")

        # "Route" tokens within each shard.
        num_experts_per_shard = self.config.num_experts // num_ep
        (
            x,
            sorted_selected_experts,
            weights,
            group_sizes,
            selected_experts,
            lb_loss,
            bias_updates,
            local_group_sizes,
        ) = self.permute(
            x,
            logits,
            pre_bias_logits,
            self.config.use_custom_sort_vjp,
            roll_to_expert_id=num_experts_per_shard * expert_shard_id,
            rngs=rngs,
            input_ids=input_ids,
            dispatch_x_is_local=chunk_dispatch,
            saved_sort=saved_sort,
            sort_save_cell=sort_save_cell,
        )

      else:
        if saved_sort is not None or sort_save_cell is not None:
          raise ValueError("moe_save_sort_indices requires use_ring_of_experts=True.")
        (
            x,
            sorted_selected_experts,
            weights,
            group_sizes,
            selected_experts,
            lb_loss,
            bias_updates,
            local_group_sizes,
        ) = self.permute(x, logits, pre_bias_logits, self.config.use_custom_sort_vjp, rngs, input_ids=input_ids)

        if num_ep > 1:
          batch_axis = self._expert_parallelism_name if is_batch_sharded_by_expert else "data"
          # get group sizes for all shards
          local_expert_size = self.config.num_experts // num_ep
          reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
          global_group_sizes = group_sizes

          if is_batch_sharded_by_expert:
            all_shards_group_sizes = jax.lax.all_gather(reshaped_group_sizes, axis_name=batch_axis)
            input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
                all_shards_group_sizes,
                expert_shard_id,
                num_ep,
            )

            buffer_size = self.get_ragged_buffer_size(
                jnp.shape(x)[0],
                num_ep,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                self.config.ragged_buffer_factor,
            )
            output_shape = jax.lax.empty((buffer_size, self.moe_expert_input_dim), dtype=x.dtype)

            x = jax.lax.ragged_all_to_all(
                x,
                output_shape,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name=self._expert_parallelism_name,
            )
            global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=self._expert_parallelism_name)
            x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                x,
                global_group_sizes,
                local_expert_size,
                shard_index=expert_shard_id,
                use_custom_sort_vjp=self.config.use_custom_sort_vjp,
                use_ragged_sort=self.config.use_ragged_sort,
            )
          else:
            x, local_sorted_indices, group_sizes, selected_experts = RoutedMoE.local_permute(
                x,
                global_group_sizes[None, :],
                local_expert_size,
                shard_index=expert_shard_id,
                is_offset=True,
                global_sorted_experts=selected_experts,
                use_custom_sort_vjp=self.config.use_custom_sort_vjp,
                use_ragged_sort=self.config.use_ragged_sort,
            )

      return (
          x,
          RouteOutput(
              group_sizes=group_sizes,
              selected_experts=selected_experts,
              sorted_selected_experts=sorted_selected_experts,
              weights=weights,
              lb_loss=lb_loss,
              bias_updates=bias_updates,
              local_group_sizes=local_group_sizes,
          ),
          RouteMetadata(
              expert_shard_id=expert_shard_id,
              local_sorted_indices=local_sorted_indices,
              all_shards_group_sizes=all_shards_group_sizes,
              reshaped_group_sizes=reshaped_group_sizes,
          ),
      )

    def get_active_sharding_axes(pspec_dim_axes, tensor_dim_index):
      if pspec_dim_axes is None:
        return []
      axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else pspec_dim_axes
      active = []
      for ax in axes:
        if ax and self.mesh.shape.get(ax, 1) > 1:
          active.append((ax, tensor_dim_index))
      return active

    _ring_fp8_wag = getattr(self.config, "moe_fp8_ring_weight_ag", False) and not self.config.shard_exp_on_fsdp
    def get_wi_gmm_params():
      wi_gather_axes = []
      if weight_gather:
        if _ring_fp8_wag:
          # ring fp8 weight-AG: gather ONLY the FSDP-sharded In/embed (dim 1, the GMM contracting dim)
          wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[1], 1))
        else:
          # wi [Experts, In, Hidden] -> Gather Exp(0) and Hidden(2)
          wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[0], 0))
          wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[2], 2))
      wi_tile_size = (
          self.config.wi_tile_fwd_batch_seq,  # m (LHS batch)
          self.config.wi_tile_fwd_embed_dim,  # k  (contracting)
          self.config.wi_tile_fwd_mlp_dim,  # n (RHS batch)
          self.config.wi_tile_dlhs_batch_seq,  # m (LHS batch)
          self.config.wi_tile_dlhs_mlp_dim,  # k (contracting)
          self.config.wi_tile_dlhs_embed_dim,  # n (RHS batch)
          self.config.wi_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
          self.config.wi_tile_drhs_embed_dim,  # Called k in megablox, but this is LHS batch dim
          self.config.wi_tile_drhs_mlp_dim,  # Called n in megablox, and indeed is RHS batch dim
      )
      return wi_gather_axes, wi_tile_size

    def get_wo_gmm_params():
      wo_gather_axes = []
      if weight_gather:
        if _ring_fp8_wag:
          # ring fp8 weight-AG: gather ONLY the FSDP-sharded Out/embed (dim 2, the GMM output dim)
          wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[2], 2))
        else:
          # wo [Experts, Hidden, Out] -> Gather Exp(0) and Hidden(1)
          wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[0], 0))
          wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[1], 1))
      wo_tile_size = (
          self.config.wo_tile_fwd_batch_seq,  # m (LHS batch)
          self.config.wo_tile_fwd_mlp_dim,  # k (contracting)
          self.config.wo_tile_fwd_embed_dim,  # n (RHS batch)
          self.config.wo_tile_dlhs_batch_seq,  # m (LHS batch)
          self.config.wo_tile_dlhs_embed_dim,  # k (contracting)
          self.config.wo_tile_dlhs_mlp_dim,  # n (RHS)
          self.config.wo_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
          self.config.wo_tile_drhs_mlp_dim,  # Called k in megablox, but this is LHS batch dim
          self.config.wo_tile_drhs_embed_dim,  # Called n in megablox, and indeed is the RHS batch dim
      )
      return wo_gather_axes, wo_tile_size

    def gmm_up(x, w0, w1, w0_bias, w1_bias, gmm_fn, weight_gather, w0_scale=None, w1_scale=None):
      """Run the two up-projections (gate + up) and apply the FFN activation."""
      wi_gather_axes, wi_tile_size = get_wi_gmm_params()
      # moe_fp8_boundary_qag: build the w0/w1 QArrays at the LAST moment (raw e4m3 qvalue + per-tensor
      # scale). Force the non-prefuse path (can't concat two e4m3 tensors with different per-tensor scales).
      if self.config.prefuse_moe_weights and not _fp8q:
        # Weights are stored as (G,K,2N); w0/w1 are adjacent slices so XLA elides this concat.
        w_fused = jnp.concatenate([w0, w1], axis=-1)
        out = gmm_fn(x, w_fused, tiling=wi_tile_size, weight_gather_axes=wi_gather_axes)
        n = out.shape[-1] // 2
        layer_w0, layer_w1 = out[:, :n], out[:, n:]
        if self.get_tensor_transpose_parallelism_size() > 1:
          layer_w0 = jax.lax.psum(layer_w0, "tensor_transpose")
          layer_w1 = jax.lax.psum(layer_w1, "tensor_transpose")
        if self.config.mlp_bias:
          layer_w0 = layer_w0 + w0_bias
          layer_w1 = layer_w1 + w1_bias
        layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")
        layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
      else:
        # stop_gradient on the scale: kills the cotangent path to the replicated shard_map scale
        # input -- without it, the shard_map transpose psums the (always-zero, but opaque through
        # the gmm custom_vjp) scale ct per layer, and remat re-materializes those tiny psums in the
        # backward at the SC-offload ~21ms latency floor (the +3.7s/step cluster regression).
        _w0 = (
            qpl.QArray(qvalue=w0, scale=jax.lax.stop_gradient(w0_scale), zero_point=None, qtype=jnp.float8_e4m3fn)
            if _fp8q else w0
        )
        _w1 = (
            qpl.QArray(qvalue=w1, scale=jax.lax.stop_gradient(w1_scale), zero_point=None, qtype=jnp.float8_e4m3fn)
            if _fp8q else w1
        )
        layer_w0 = gmm_fn(
            x,
            _w0,
            tiling=wi_tile_size,
            weight_gather_axes=wi_gather_axes,
        )
        if self.get_tensor_transpose_parallelism_size() > 1:
          layer_w0 = jax.lax.psum(layer_w0, "tensor_transpose")
        if self.config.mlp_bias:
          layer_w0 = layer_w0 + w0_bias
        layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")

        layer_w1 = gmm_fn(
            x,
            _w1,
            tiling=wi_tile_size,
            weight_gather_axes=wi_gather_axes,
        )
        if self.get_tensor_transpose_parallelism_size() > 1:
          layer_w1 = jax.lax.psum(layer_w1, "tensor_transpose")
        if self.config.mlp_bias:
          layer_w1 = layer_w1 + w1_bias
        layer_w1 = adc.checkpoint_name(layer_w1, "moe_mlpwi_1")
      return self.apply_ffn_activation(layer_w0, layer_w1)

    def get_gmm_for_local_experts(x, routing, route_metadata):
      """Return a partial GMM function with preconfigured routing params."""
      num_ep = self.get_expert_parallelism_size()
      num_experts_per_shard = self.config.num_experts // num_ep
      if self.config.use_ring_of_experts and x.shape[0] < routing.sorted_selected_experts.shape[0]:
        local_group_sizes = routing.local_group_sizes
        return functools.partial(
            gmm,
            group_sizes=local_group_sizes,
            expert_assignments=routing.selected_experts,
            group_offset=0,
        )
      if self.config.use_ragged_sort and self.config.use_ring_of_experts:
        experts_start = route_metadata.expert_shard_id * num_experts_per_shard
      else:
        experts_start = 0
      return functools.partial(
          gmm,
          group_sizes=routing.group_sizes,
          expert_assignments=routing.selected_experts,
          group_offset=experts_start,
      )

    def unsort_output_and_ra2a(intermediate_output, routing, route_metadata, output_shape, is_batch_sharded_by_expert):
      """Unsort tokens and return them to original shards using ragged all-to-all."""
      if is_batch_sharded_by_expert:
        # locally unpermute back to the original order
        if self.config.use_ragged_sort:
          # Mirror the ragged-prefix gather used in `local_permute`. The
          # un-permute can use the same valid-prefix length because the
          # routed token count is identical for forward and backward.
          valid_end = jnp.sum(routing.group_sizes).astype(jnp.int32)
          local_output = a2a_ragged_unsort(
              intermediate_output,
              jnp.argsort(route_metadata.local_sorted_indices),  # pylint: disable=undefined-variable
              valid_end,
              use_single_sparsecore=self.config.ragged_sort_use_single_sparsecore,
          )
        else:
          local_output = _sort_activations(
              intermediate_output,
              jnp.argsort(route_metadata.local_sorted_indices),
              self.config.use_custom_sort_vjp,
          )

        input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
            jnp.transpose(route_metadata.all_shards_group_sizes),
            route_metadata.expert_shard_id,
            self.get_expert_parallelism_size(),
        )
        return jax.lax.ragged_all_to_all(
            local_output,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=self._expert_parallelism_name,
        )

      # If batch is replicated across EP shards then each shard should send
      # 0..local_shard_size data to the other shards and receive the
      # local_shard data from all of the other shards using ragged_all_to_all.
      input_offsets, send_sizes, output_offsets, recv_sizes = RoutedMoE.get_all_to_all_params(
          route_metadata.reshaped_group_sizes,
          route_metadata.expert_shard_id,
          self.get_expert_parallelism_size(),
          is_batch_sharded=False,
      )
      return jax.lax.ragged_all_to_all(
          intermediate_output,
          output_shape,
          input_offsets,
          send_sizes,
          output_offsets,
          recv_sizes,
          axis_name=self._expert_parallelism_name,
      )

    def _moe_body(
        x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, sharded_input_ids, rngs,
        w0_scale=None, w1_scale=None, wo_scale=None, saved_sort=None, sort_save_cell=None,
    ):
      batch_size, sequence_length, _ = x.shape

      if self.config.use_fused_a2a:
        # PHASE B: replace the a2a route->gmm->combine with the vendored fused K1/K2 a2a kernel.
        # up->down ONLY (kernel has no gate/SwiGLU yet) => loss is WRONG; this is the PERF signal.
        # _moe_body runs inside sparse_matmul's shard_map (expert axis present) so the fused
        # layer's collectives work without a nested shard_map. w1=up, wo=down (gate w0 skipped).
        from maxtext.kernels.a2a_fused.fused_layer_vjp import make_fused_moe_layer

        _D = x.shape[-1]
        _k = self.num_experts_per_tok
        _weights, _sel = self.get_topk(logits, pre_bias_logits, rngs, sharded_input_ids)
        _T = batch_size * sequence_length
        _blk = 512
        _cap = ((_T * _k + _blk - 1) // _blk) * _blk  # align_up(T*k, blk); dropless CAP
        _mn = self.mesh.axis_names
        _ms = dict(zip(self.mesh.axis_names, self.mesh.devices.shape))
        _layer = make_fused_moe_layer(
            ep=self.get_expert_parallelism_size(),
            num_experts=self.config.num_experts,
            cap_rows=_cap,
            blk=_blk,
            ep_axis=self._expert_parallelism_name,
            mesh_axis_names=_mn,
            mesh_shape=_ms,
            recompute_residuals=True,
            dw_impl="tgmm",
            self_last=True,
        )
        _out = _layer(
            x.reshape(_T, _D).astype(jnp.bfloat16),
            _sel.reshape(_T, _k).astype(jnp.int32),
            _weights.reshape(_T, _k).astype(jnp.float32),
            w1.astype(jnp.bfloat16),
            wo.astype(jnp.bfloat16),
        )
        return _out.reshape(batch_size, sequence_length, _D).astype(x.dtype), None, None

      x, routing, route_metadata = route(
          x, logits, pre_bias_logits, rngs, input_ids=sharded_input_ids,
          saved_sort=saved_sort, sort_save_cell=sort_save_cell,
      )
      # (moe_x_sorted option A: the save tag lives on the PRE-duplication gathered tokens inside
      # route()'s dispatch block -- NOT here on the post-sort x, whose topk-8 duplication made the
      # save 229GB/compile-OOM. The sort re-runs in the backward from the saved gathered tokens.)

      if self.config.mlp_bias:
        w0_bias, w1_bias, wo_bias = self.transform_bias(routing.selected_experts, w0_bias, w1_bias, wo_bias)

      gmm_fn = get_gmm_for_local_experts(x, routing, route_metadata)
      intermediate_layer = gmm_up(x, w0, w1, w0_bias, w1_bias, gmm_fn, weight_gather, w0_scale, w1_scale)

      wo_gather_axes, wo_tile_size = get_wo_gmm_params()
      # moe_fp8_cv_weight_ag (_fp8wo): wo arrives as an e4m3 qvalue gathered by the cv-gather; build
      # its QArray here (global [1,1,embed] scale -- identical on every shard, no consistency issue).
      # moe_fp8_boundary_qag keeps wo bf16 (per-output-channel over its embed GSPMD-gather is unsound).
      # stop_gradient on the scale: no ct path -> no per-layer boundary psum (see gmm_up).
      _wo = (
          qpl.QArray(qvalue=wo, scale=jax.lax.stop_gradient(wo_scale), zero_point=None, qtype=jnp.float8_e4m3fn)
          if _fp8wo else wo
      )
      intermediate_output = gmm_fn(
          intermediate_layer,
          _wo,
          tiling=wo_tile_size,
          weight_gather_axes=wo_gather_axes,
      )
      if self.get_tensor_parallelism_size() > 1:
        intermediate_output = jax.lax.psum_scatter(
            intermediate_output, self._tensor_parallelism_name, scatter_dimension=1, tiled=True
        )
      if self.config.mlp_bias:
        intermediate_output = intermediate_output + wo_bias
      intermediate_output = adc.checkpoint_name(adc.checkpoint_name(intermediate_output, "mlpwo"), "moe_mlpwo")

      if (
          self.config.use_ring_of_experts
          and self.config.decouple_combine_rs_chunks > 1
          and use_chunked_combine
          and isinstance(self._expert_parallelism_name, str)
      ):
        # DECOUPLED chunked combine->RS: the GMM ran FULL above; here we chunk ONLY combine+RS so
        # each chunk's reduce-scatter hides under the next chunk's combine (validated v7x). The
        # expert-sorted GMM output is read WHOLE per chunk; only the token (output) axis is chunked.
        # See chunked_ring_combine_reduce_scatter for the expert->token handling + the permute trick.
        # `use_chunked_combine` is False on the moe_handwritten_bwd RECOMPUTE path (deepseek.py
        # fused_bwd), which differentiates the un-chunked combine instead (forward-only chunking).
        drs_fn = None
        if self.config.moe_direct_rs and self._expert_parallelism_name == "expert":
          _mesh = self.mesh

          _splash_off_sg = self._splash_offload_sched_group()

          def drs_fn(x, chunk_idx):
            # Per-chunk DISTINCT collective_id: the chunk RSs can be concurrently in flight
            # (that overlap is the whole lever), so they must not share a barrier semaphore.
            # sched_group tags the per-chunk transpose all-gathers for the splash-offload overlap
            # (only relevant when moe_chunked_combine_in_remat differentiates the chunked combine).
            return _direct_reduce_scatter(x, _mesh, "expert", 7 + chunk_idx, _splash_off_sg)

        ag_fn = None
        if self.config.moe_direct_combine_ag and self._expert_parallelism_name == "expert":
          _mesh_ag = self.mesh

          def ag_fn(g_out):
            # moe_direct_combine_ag (BACKWARD combine-cotangent only, gated): all-gather the whole
            # contiguous g_out (bf16[num_tokens, hidden], == all-gather.626, the big exposed backward
            # gather) with the direct-to-owner TC Pallas kernel instead of the XLA collective, so it
            # rides the TensorCore ICI DMAs -- NOT the SparseCore all-gather-offload queue that
            # serializes lax.all_gather behind the SC combines -- letting XLA overlap the two
            # (different engines). Numerically == lax.all_gather (verified in isolation); its
            # custom_vjp gives the same psum_scatter transpose the collective would.
            return _direct_all_gather(g_out, _mesh_ag, "expert", _DIRECT_COMBINE_AG_COLLECTIVE_ID)

        output = chunked_ring_combine_reduce_scatter(
            intermediate_output,
            routing.group_sizes,
            routing.sorted_selected_experts,
            self.num_experts_per_tok,
            self.config.num_experts // self.get_expert_parallelism_size(),
            self._expert_parallelism_name,
            jnp.ravel(routing.weights).astype(jnp.float32),
            self.get_expert_parallelism_size(),
            self.config.decouple_combine_rs_chunks,
            return_first_combine_token=emit_combine_token,
            reduce_scatter_fn=drs_fn,
            all_gather_fn=ag_fn,
            enforce_gather_fallback=self.config.ragged_gather_fallback,
            enforce_gather_reduce_fallback=self.config.ragged_gather_reduce_fallback,
            gather_flops_override=self.config.ragged_gather_cost_estimate_flops,
            gather_reduce_flops_override=self.config.ragged_gather_reduce_cost_estimate_flops,
            gather_bytes_accessed_override=self.config.ragged_gather_cost_estimate_bytes_accessed,
            gather_reduce_bytes_accessed_override=self.config.ragged_gather_reduce_cost_estimate_bytes_accessed,
        )
        if emit_combine_token:
            output, combine_token = output
        output = output.reshape(
            -1, sequence_length, self.moe_expert_input_dim // self.get_tensor_parallelism_size()
        ).astype(self.dtype)
        if emit_combine_token:
          return output, routing.lb_loss, routing.bias_updates, combine_token
        return output, routing.lb_loss, routing.bias_updates

      if self.config.use_ring_of_experts:
        # Unsort and deduplicate the outputs locally.
        output = self.unpermute(
            intermediate_output,
            routing.sorted_selected_experts,
            routing.weights,
            batch_size=batch_size,
            sequence_length=sequence_length,
            use_custom_sort_vjp=self.config.use_custom_sort_vjp,
            group_sizes=routing.group_sizes,
        )

        # Sum up the partial outputs across the expert shards.
        output = jnp.reshape(
            output, (-1, sequence_length, self.moe_expert_input_dim // self.get_tensor_parallelism_size())
        )
        if self.config.moe_direct_rs and self._expert_parallelism_name == "expert":
          # Direct-to-owner Pallas RS (TC) so XLA can overlap its ICI DMA under the SC combine,
          # instead of the psum_scatter parking behind the SC-offload queue. == psum_scatter
          # (rel 0.004 bf16 reduce-order). Same gating as the old-branch wiring (plain axis).
          # sched_group (moe_splash_offload_scheduling_group): this un-chunked combine runs in the
          # manbwd RECOMPUTE; tagging its transpose all-gather (_drs_bwd == all-gather.626) lets the
          # scheduler overlap it with the co-tagged splash host restore copies. None otherwise.
          output = _direct_reduce_scatter(output, self.mesh, "expert", 7, self._splash_offload_sched_group())
        elif getattr(self.config, "moe_ring_combine_rs", False) and self._expert_parallelism_name == "expert":
          # moe_ring_combine_rs: BOTH directions on TC ring kernels -- forward = the ring RS twin
          # (== psum_scatter rel=0), backward = the ring cotangent AG. No remat save needed: the
          # dump census shows the combine is NOT re-run in the backward remat (the recompute stops
          # at the mlpwo/moe_mlpwo save before it), so the fwd Pallas kernel never fires in a
          # rematted region (the remat+Pallas-DMA rule holds without a save).
          if not getattr(self.config, "moe_ring_cotangent_ag", False):
            raise ValueError("moe_ring_combine_rs requires moe_ring_cotangent_ag=True.")
          output = _ring_combine_rs(
              output, self.mesh, "expert", _RING_RS_COLLECTIVE_ID, _RING_CT_AG_COLLECTIVE_ID
          )
        elif getattr(self.config, "moe_ring_cotangent_ag", False) and self._expert_parallelism_name == "expert":
          # moe_ring_cotangent_ag: forward = the SAME plain psum_scatter (untouched); ONLY the
          # backward cotangent all-gather moves onto the TC ring kernel (see _ring_ct_reduce_scatter).
          output = _ring_ct_reduce_scatter(output, self.mesh, "expert", _RING_CT_AG_COLLECTIVE_ID)
        else:
          output = jax.lax.psum_scatter(output, self._expert_parallelism_name, scatter_dimension=0, tiled=True)
        return output, routing.lb_loss, routing.bias_updates

      if self.get_expert_parallelism_size() > 1:
        original_inputs_first_dim = batch_size * sequence_length * self.config.num_experts_per_tok
        if routing.sorted_selected_experts.shape[0] != original_inputs_first_dim:
          raise ValueError("original_inputs_first_dim does not match the original tensor" " shape!")
        output_shape = jax.lax.empty(
            (
                original_inputs_first_dim,
                self.moe_expert_input_dim // self.get_tensor_parallelism_size(),
            ),
            dtype=intermediate_output.dtype,
        )

        intermediate_output = unsort_output_and_ra2a(
            intermediate_output,
            routing,
            route_metadata,
            output_shape,
            is_batch_sharded_by_expert,
        )

      output = self.unpermute(
          intermediate_output,
          routing.sorted_selected_experts,
          routing.weights,
          batch_size=batch_size,
          sequence_length=sequence_length,
          use_custom_sort_vjp=self.config.use_custom_sort_vjp,
          group_sizes=routing.group_sizes,
      )

      return output, routing.lb_loss, routing.bias_updates

    # moe_save_sort_indices: pspecs for the routing bundle crossing the shard_map boundary.
    # All four tensors are computed from the EP-all-gathered logits -> REPLICATED over the
    # expert axis, sharded over the remaining batch axes (so the fwd-out -> residual -> bwd-in
    # round trip is a pure slice, no collectives). group_sizes ([num_experts] per shard, with
    # per-(data/fsdp)-shard VALUES) crosses with a leading size-1 batch-carrier axis.
    save_or_load_routing = save_routing or (saved_routing is not None)
    routing_bundle_specs = None
    if save_or_load_routing:
      if not (self.config.use_ring_of_experts and self.config.use_ragged_sort):
        raise ValueError("moe_save_sort_indices requires use_ring_of_experts=True and use_ragged_sort=True.")
      _ep_axis = self._expert_parallelism_name

      def _drop_ep(entry):
        if isinstance(entry, (tuple, list)):
          kept = tuple(a for a in entry if a != _ep_axis)
          return kept if kept else None
        return None if entry == _ep_axis else entry

      _saved_batch = _drop_ep(gate_logits_pspec[0] if len(gate_logits_pspec) > 0 else None)
      _saved_seq = gate_logits_pspec[1] if len(gate_logits_pspec) > 1 else None
      routing_chunk_specs = (
          P(_saved_batch, _saved_seq, None),  # top_k_indices [b, s_chunk, k]
          P(_saved_batch),  # token_indices_sorted [b*s_chunk*k]
          P(_saved_batch, None),  # group_sizes [1, num_experts] (leading batch-carrier axis)
          P(_saved_batch),  # topk_argsort_revert_indices [b*s_chunk*k]
      )
      n_routing_chunks = self.config.num_moe_token_chunks if self.config.num_moe_token_chunks > 1 else 1
      routing_bundle_specs = (routing_chunk_specs,) * n_routing_chunks

    def _pack_routing_chunk(bundle):
      sel, tis, gs, rev = bundle
      return (sel, tis, gs[None], rev)

    def _unpack_routing_chunk(bundle):
      sel, tis, gs, rev = bundle
      return (sel, tis, gs[0], rev)

    # moe_fp8_boundary_qag: the weight args carry the e4m3 QVALUE (keep the weight pspec -> GSPMD
    # boundary-gathers only e4m3); the per-tensor SCALES ride separate replicated args; the QArray is
    # reconstructed inside the body (no bf16 dequant between gather and gmm consumer -> no elision).
    _fp8bq = getattr(self.config, "moe_fp8_boundary_qag", False)
    _fp8cv = getattr(self.config, "moe_fp8_cv_weight_ag", False)
    # moe_fp8_cv_weight_ag: the weights arrive as qwix QArrays (e4m3 qvalue STILL STORAGE-SHARDED on
    # fsdp-embed + replicated [1,1,n] f32 scale). Split them here, BEFORE the shard_map decorator, and
    # OVERRIDE the weight in_specs to the STORAGE sharding: the e4m3 all-gather then happens INSIDE
    # the body (top of sparse_matmul_route_and_compute). Structure rationale (cluster round-2 profile):
    # with a replicated-in weight, the shard_map transpose ARs the weight grad and the reshard slices
    # it -- XLA failed to fuse that into one reduce-scatter (two slow RSs @ 22-38 GB/s + extra AR,
    # +0.35s/step). With a SHARDED-in weight + in-body gather, autodiff transposes the gather into ONE
    # direct psum_scatter to storage sharding (the baseline's efficient 235 GB/s form), and the
    # boundary adds no extra reduction (the input is varying, not replicated) -> no x128 overcount.
    _fp8cv_in = _fp8cv and isinstance(w0_kernel, qpl.QArray)
    if _fp8cv_in:
      _w0sc, w0_kernel = w0_kernel.scale, w0_kernel.qvalue
      _w1sc, w1_kernel = w1_kernel.scale, w1_kernel.qvalue
      _wosc, wo_kernel = wo_kernel.scale, wo_kernel.qvalue
      w0_pspec = self._logical_to_mesh_axes(self.wi_kernel_axes)
      w1_pspec = self._logical_to_mesh_axes(self.wi_kernel_axes)
      wo_pspec = self._logical_to_mesh_axes(self.wo_kernel_axes)
    # moe_fp8_boundary_qag: the w0/w1 fp8 scale is [1,1,mlp] (per-mlp-channel, shared across experts),
    # so it is REPLICATED -> P(). (A per-expert [exp,1,mlp] scale would need expert-axis sharding AND
    # a stock gmm_v2 backward fix; deferred -- see _q_boundary.)
    _w0sc_pspec = _w1sc_pspec = P()

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            input_partition_pspec,
            gate_logits_pspec,
            pre_bias_logits_pspec,
            w0_pspec,
            w1_pspec,
            wo_pspec,
            w0_bias_pspec,
            w1_bias_pspec,
            wo_bias_pspec,
            decoder_tokens_pspec,
            P(),  # Replicate the input key
            _w0sc_pspec,  # w0 fp8 qvalue scale (moe_fp8_boundary_qag, per-expert-channel sharded)
            _w1sc_pspec,  # w1 fp8 qvalue scale (per-expert-channel sharded)
            P(),  # wo scale unused (wo stays bf16)
            routing_bundle_specs if saved_routing is not None else None,
        ),
        out_specs=(
            self._logical_to_mesh_axes((batch_logical_axis, "activation_norm_length", "activation_embed")),
            P(),  # Handle None or replicate the output
            P(),  # Handle None or replicate the output
        )
        # [1, 1] combine SCHEDULING token (moe_shared_after_combine). P() types it replicated
        # although each device holds its own shard's value -- safe because the token's VALUE is
        # never consumed (it only carries a scheduling dependency into an optimization_barrier)
        # and no resharding/collective is ever inserted on it. Requires check_vma=False, which
        # is forced anyway on the ring-of-experts path (see base.yml note on check_vma).
        + ((P(),) if emit_combine_token else ())
        # moe_save_sort_indices: per-chunk saved routing bundles (appended LAST).
        + ((routing_bundle_specs,) if save_routing else ()),
        check_vma=self.config.check_vma,
    )
    def sparse_matmul_route_and_compute(
        x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, sharded_input_ids, rngs,
        w0_scale, w1_scale, wo_scale, saved_routing_in
    ):
      # moe_fp8_cv_weight_ag (_fp8cv_in): w0/w1/wo arrive STORAGE-SHARDED e4m3 qvalues (fsdp on the
      # embed dim). Gather them here, ONCE, INSIDE the body: the explicit e4m3 lax.all_gather is
      # elision-proof, and its autodiff transpose is ONE direct psum_scatter of the bf16 weight grad
      # to storage sharding (the efficient single-RS form; no boundary AR, no x128 overcount).
      # Gathered once, reused across all chunks. The QArray is still built at the LAST moment before
      # each gmm_fn call; the scales (w0_scale/w1_scale/wo_scale) thread alongside.
      if _fp8cv_in:
        w0 = jax.lax.all_gather(w0, "fsdp", axis=1, tiled=True)  # [exp, embed_full, mlp]
        w1 = jax.lax.all_gather(w1, "fsdp", axis=1, tiled=True)
        wo = jax.lax.all_gather(wo, "fsdp", axis=2, tiled=True)  # [exp, mlp, embed_full]
      # (moe_fp8_boundary_qag: w0/w1 arrive boundary-gathered e4m3; same last-moment QArray plumbing.)
      # The expert weights (w0/w1/wo) are all-gathered over FSDP once at this
      # shard_map entry (implicitly, via the `embed_tensor_transpose` pspec which
      # drops fsdp -> GSPMD inserts the boundary all-gather) and reused across all
      # chunks of the ring-of-experts pipeline below.
      n_chunks = self.config.num_moe_token_chunks
      if n_chunks <= 1 or not self.config.use_ring_of_experts:
        cell = {} if save_routing else None
        saved_c = _unpack_routing_chunk(saved_routing_in[0]) if saved_routing_in is not None else None
        result = _moe_body(
            x, logits, pre_bias_logits, w0, w1, wo, w0_bias, w1_bias, wo_bias, sharded_input_ids, rngs,
            w0_scale=w0_scale, w1_scale=w1_scale, wo_scale=wo_scale,
            saved_sort=saved_c, sort_save_cell=cell,
        )
        if save_routing:
          return tuple(result) + ((_pack_routing_chunk(cell["sort_bundle"]),),)
        return result

      # Chunked ring-of-experts pipeline: split the per-shard tokens along the
      # sequence dim into `n_chunks` data-independent chunks. Each chunk runs the
      # full route -> GMM -> combine path; with no barrier between them XLA is
      # free to overlap chunk (c+1)'s EP all-gather and chunk (c-1)'s
      # reduce-scatter with chunk c's GMM compute. Token routing is per-token, so
      # the main (lm) output is identical to n_chunks=1; only the aggregate
      # load-balance loss / bias updates are averaged across chunks.
      seq_len = x.shape[1]
      if seq_len % n_chunks != 0:
        raise ValueError(f"num_moe_token_chunks={n_chunks} must evenly divide the MoE sequence length {seq_len}.")
      chunk = seq_len // n_chunks
      outs, lb_losses, bias_updates_list, routing_bundles = [], [], [], []
      _prev = None
      for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        x_c = x[:, sl, :]
        # Diagnostic: fence each chunk's input on the previous chunk's output so XLA
        # cannot interleave/fuse the chunks -- forces sequential pipelining. Math is
        # unchanged (the barrier is identity), so loss stays bit-exact.
        if self.config.moe_chunk_barrier and _prev is not None:
          x_c, _prev = jax.lax.optimization_barrier((x_c, _prev))
        cell = {} if save_routing else None
        saved_c = _unpack_routing_chunk(saved_routing_in[c]) if saved_routing_in is not None else None
        out_c, lb_c, bu_c = _moe_body(
            x_c,
            logits[:, sl, :],
            None if pre_bias_logits is None else pre_bias_logits[:, sl, :],
            w0,
            w1,
            wo,
            w0_bias,
            w1_bias,
            wo_bias,
            None if sharded_input_ids is None else sharded_input_ids[:, sl],
            rngs,
            w0_scale=w0_scale,
            w1_scale=w1_scale,
            wo_scale=wo_scale,
            saved_sort=saved_c,
            sort_save_cell=cell,
        )
        if self.config.moe_chunk_barrier:
          _prev = out_c
        outs.append(out_c)
        lb_losses.append(lb_c)
        bias_updates_list.append(bu_c)
        if save_routing:
          routing_bundles.append(_pack_routing_chunk(cell["sort_bundle"]))
      output = jnp.concatenate(outs, axis=1)
      lb_loss = None if lb_losses[0] is None else sum(lb_losses) / n_chunks
      bias_updates = None if bias_updates_list[0] is None else sum(bias_updates_list) / n_chunks
      if save_routing:
        return output, lb_loss, bias_updates, tuple(routing_bundles)
      return output, lb_loss, bias_updates

    if self.config.moe_fsdp_use_two_stage_all_gather:
      # Unshard on fsdp axis
      w0_kernel = self._maybe_shard_with_logical(w0_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp"))
      w1_kernel = self._maybe_shard_with_logical(w1_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp"))

      # Unshard on fsdp_transpose axis
      wo_kernel = self._maybe_shard_with_logical(wo_kernel, ("exp_with_fsdp", "mlp", "embed_tensor_transpose"))

      # Make sure XLA does not optimize by combining above All-Gather to unshard
      # on FSDP axis and the subsequent unshard on fsdp_transpose axis
      w0_kernel = jax.lax.optimization_barrier(w0_kernel)
      w1_kernel = jax.lax.optimization_barrier(w1_kernel)
      wo_kernel = jax.lax.optimization_barrier(wo_kernel)

      # Unshard on both fsdp and fsdp_transpose transpose
      w0_kernel = self._maybe_shard_with_logical(w0_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp_no_fsdp"))
      w1_kernel = self._maybe_shard_with_logical(w1_kernel, ("exp_with_fsdp", "embed_tensor_transpose", "mlp_no_fsdp"))
      wo_kernel = self._maybe_shard_with_logical(wo_kernel, ("exp_with_fsdp", "mlp_no_fsdp", "embed_tensor_transpose"))

    if self.get_tensor_transpose_parallelism_size() > 1:
      input_axes = (batch_logical_axis, "activation_norm_length", "activation_embed")
    else:
      input_axes = (batch_logical_axis, "activation_norm_length", None)

    gate_logits_axes = (batch_logical_axis, "activation_norm_length", None)
    # NOTE: deepseek2 has a different pattern
    if self.config.model_name.startswith(("deepseek3", "deepseek4")):
      pre_bias_logits_axes = (batch_logical_axis, "activation_norm_length", None)
    else:
      pre_bias_logits_axes = None

    inputs = self._maybe_shard_with_logical(inputs, input_axes)
    gate_logits = self._maybe_shard_with_logical(gate_logits, gate_logits_axes)
    pre_bias_logits = self._maybe_shard_with_logical(pre_bias_logits, pre_bias_logits_axes)

    w0_kernel = self._maybe_shard_with_pspec(w0_kernel, w0_pspec)
    w1_kernel = self._maybe_shard_with_pspec(w1_kernel, w1_pspec)
    wo_kernel = self._maybe_shard_with_pspec(wo_kernel, wo_pspec)
    if w0_bias is not None:
      w0_bias = self._maybe_shard_with_pspec(w0_bias, w0_bias_pspec)
    if w1_bias is not None:
      w1_bias = self._maybe_shard_with_pspec(w1_bias, w1_bias_pspec)
    if wo_bias is not None:
      wo_bias = self._maybe_shard_with_pspec(wo_bias, wo_bias_pspec)

    # moe_fp8_boundary_qag: quantize w0/w1 to (e4m3 qvalue, DYNAMIC PER-OUTPUT-CHANNEL scale) at the
    # shard_map boundary. w0/w1 are [exp, embed, mlp], gathered over embed (fsdp); the output channel
    # is the mlp axis, so scale = max(|w|, axis=embed)/448 -> [exp,1,mlp]. That max over the
    # fsdp-sharded embed all-reduces a small [exp,1,mlp] tensor (cheap), and the scale is replicated
    # along the embed gather axis -> gathering the e4m3 qvalue stays sound. wo is [exp, mlp, embed];
    # its output channel IS embed = the gather axis, so per-output-channel + gather-over-embed is
    # unsound -> wo STAYS bf16 (never quantized). Pass the qvalue as the weight arg (keeps the pspec
    # -> boundary-gathers e4m3) + the scale as a replicated arg; the QArray is rebuilt before gmm_fn.
    def _q_boundary(w):
      # DYNAMIC PER-OUTPUT-CHANNEL scale over the mlp axis, SHARED across the local experts: max over
      # (expert axis 0, embed axis 1) -> [1,1,mlp]. Per-EXPERT [exp,1,mlp] is more granular but breaks
      # stock gmm_v2 backward (_dlhs_scale_grad_by_rhs_scale repeats the per-expert scale by the GLOBAL
      # group_sizes[256] vs the 32 local experts). [1,1,mlp] hits the shared-scale branch (no repeat),
      # stays replicated along the embed gather axis (sound), and is still per-channel dynamic (not
      # per-tensor). The max over the fsdp-sharded embed all-reduces a tiny [1,1,mlp] tensor (cheap).
      sc = jax.lax.stop_gradient(
          jnp.max(jnp.abs(w), axis=(0, 1), keepdims=True).astype(jnp.float32) / 448.0 + 1e-20)  # [1,1,mlp]
      qv = _ste_quant(w, sc)  # STE custom_vjp: e4m3 fwd, bf16 d w = g/sc bwd (no e4m3 gradient)
      # Pin the e4m3 qvalue with an optimization_barrier so XLA's algebraic simplifier cannot sink the
      # bf16->e4m3 convert PAST the GSPMD boundary all-gather (which would gather bf16). REQUIRES
      # xla_tpu_aggressive_opt_barrier_removal=false at runtime, else the barrier is deleted first.
      qv = jax.lax.optimization_barrier(qv)
      return qv, sc
    if _fp8cv_in:
      pass  # scales already split from the pregathered QArrays above
    elif _fp8bq:
      w0_kernel, _w0sc = _q_boundary(w0_kernel)
      w1_kernel, _w1sc = _q_boundary(w1_kernel)
      _wosc = jnp.float32(1.0)  # wo stays bf16 (per-output-channel over the embed gather axis is unsound)
    else:
      _w0sc = _w1sc = _wosc = jnp.float32(1.0)
    # Body-level fp8 gates (late-bound closures read these at trace time inside the shard_map body):
    # _fp8q -> build the w0/w1 QArray at the gmm call sites; _fp8wo -> wo too (cv path only).
    _fp8q = _fp8bq or _fp8cv_in
    _fp8wo = _fp8cv_in
    result = sparse_matmul_route_and_compute(
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        w0_bias,
        w1_bias,
        wo_bias,
        input_ids,
        self.rngs,
        _w0sc,
        _w1sc,
        _wosc,
        saved_routing,
    )
    if return_combine_token and not emit_combine_token:
      # Insert the None combine token in its slot (before the routing bundle, if any).
      if save_routing:
        result = result[:-1] + (None,) + result[-1:]
      else:
        result = result + (None,)
    return result

  def reshape_and_update_weights(self, weights, indices):
    """reshape and update weights."""
    # input of weights and indices: (batch_size, seq_len, num_experts_per_tok)
    # output of updated weights: (batch_size, seq_len, num_experts)
    update_weights = jnp.zeros((weights.shape[0], weights.shape[1], self.num_experts), dtype=self.dtype)
    index_update = (
        self._maybe_shard_with_logical(
            jnp.arange(weights.shape[0])[:, None, None],
            ("activation_batch", None, None),
        ),
        self._maybe_shard_with_logical(jnp.arange(weights.shape[1])[:, None], ("activation_length", None)),
        indices,
    )
    weight_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    update_weights = update_weights.at[index_update].set(weights, out_sharding=weight_sharding)
    return update_weights

  def get_context_partition_and_sub_seq(self, seq_len):
    cp = self.get_context_autoregressive_parallelism_size()
    if seq_len % cp != 0:
      cp = 1
    sub_seq = seq_len // cp
    return cp, sub_seq

  def generate_masks_subgroup(self, top_k_indices, softmax_probs):
    """Subgroup mask generation for inference only."""
    # calculate
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape
    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    # Break sequence into subsequences (groups) of tokens, and route only within
    # each group.
    top_k_indices = jnp.reshape(top_k_indices, (batch_size, cp, sub_seq, top_k_indices.shape[2]))

    tokens_per_batch = sub_seq * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to
    # expert [0, 1] & [1, 3],
    # then expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes
    # [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of
    # updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, cp, sub_seq * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = self._maybe_shard_with_logical(expert_mask_fused, ("activation_batch", None, None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=2)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = self._maybe_shard_with_logical(
        expert_token_count,
        ("activation_batch", "activation_norm_length", None, None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=3)

    # reshape & update weights
    softmax_probs = jnp.reshape(
        softmax_probs,
        ((batch_size, cp, sub_seq, self.num_experts)),
    )
    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, cp, sub_seq, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=3) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is
    # (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    # ici_context_parallelism
    dispatch_mask = jnp.reshape(
        dispatch_mask,
        (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch),
    )
    combine_mask = jnp.reshape(
        combine_mask,
        (batch_size, cp, sub_seq, self.num_experts, expert_capacity_per_batch),
    )

    return dispatch_mask, combine_mask

  def generate_masks(self, top_k_indices, softmax_probs):
    """Generate masks."""
    # calculate
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    batch_size, seq_len, _ = top_k_indices.shape

    tokens_per_batch = seq_len * self.num_experts_per_tok
    # this is to avoid expert_capacity_per_batch = 0
    expert_capacity_per_batch = int(
        max(
            math.ceil(tokens_per_batch / self.num_experts) * self.config.capacity_factor,
            self.config.capacity_factor,
        )
    )
    max_logging.log("Applying potential token dropping with a batch expert_capacity of" f" {expert_capacity_per_batch}")

    # calculate expert mask and drop tokens if needed
    # shape of output expert mask: (batch, sequence, num_experts_per_tok)
    #
    # A small example:
    # give num_experts=4 & num_experts_per_tok=2, and two tokens are routed to
    # expert [0, 1] & [1, 3],
    # then expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 1, 0, 0],[0, 0, 0, 1]]]],
    # after cumsum, expert_token_count becomes
    # [[[[1, 0, 0, 0],[1, 1, 0, 0]], [[1, 2, 0, 0],[1, 2, 0, 1]]]],
    # if we set expert_capacity=1,
    # trunc_expert_mask becomes
    # [[[[1, 0, 0, 0],[0, 1, 0, 0]], [[0, 0, 0, 0],[0, 0, 0, 1]]]],
    # so the 2nd token for expert #1 ([0, 1] & [1, 3]) is dropped, output of
    # updated_expert_mask is [[[1, 1],[0, 1]]].
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    expert_mask_fused = jnp.reshape(
        expert_mask,
        (batch_size, seq_len * self.num_experts_per_tok, self.num_experts),
    )
    expert_mask_fused = self._maybe_shard_with_logical(expert_mask_fused, ("activation_batch_moe", None, None))
    expert_token_count_fused = jnp.cumsum(expert_mask_fused, axis=1)
    expert_token_count = jnp.reshape(
        expert_token_count_fused,
        ((batch_size, seq_len, self.num_experts_per_tok, self.num_experts)),
    )
    expert_token_count = self._maybe_shard_with_logical(
        expert_token_count,
        ("activation_batch", "activation_norm_length", None, None),
    )
    trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
    combined_expert_mask = jnp.sum(trunc_expert_mask, axis=2)

    softmax_probs *= combined_expert_mask

    # calculate token position in expert capacity dimension
    expert_token_position_fused = expert_mask_fused * expert_token_count_fused
    expert_token_position = jnp.reshape(
        expert_token_position_fused,
        (batch_size, seq_len, self.num_experts_per_tok, self.num_experts),
    )
    combined_expert_token_position = jnp.sum(expert_token_position, axis=2) * combined_expert_mask
    expert_token_position_in_capacity = jax.nn.one_hot(
        combined_expert_token_position,
        num_classes=expert_capacity_per_batch + 1,
        dtype=jnp.int32,
    )

    # shape of combine_mask is
    # (batch_size, seq_len, num_experts, expert_capacity_per_batch + 1),
    # and cut 0-dimension which is always 0
    combine_mask = softmax_probs[..., None] * expert_token_position_in_capacity
    combine_mask = combine_mask[..., 1:]
    dispatch_mask = combine_mask.astype(bool)

    return dispatch_mask, combine_mask

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits) -> jax.Array:
    """Compute the load balance loss."""
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(
      self,
      rhs_mesh_axes: Tuple[Optional[str], ...] = (),
      einsum_name: str | None = None,
  ):
    """Get the Einstein summation."""

    # the check is to prevent aqteinsum as einsum op for dispatch and combine
    # einsums in ase when capacity_factor > 0
    # this is necessary to load pre-quantized weights in case of inference
    if self.config.model_call_mode == "inference" and einsum_name in (
        DISPATCH,
        COMBINE,
    ):
      return jnp.einsum

    if self.quant:

      def aqt_einsum(*args, **kwargs):  # pylint: disable=unused-argument
        # simply skip kwargs, since aqt einsum doesn't support any kwargs
        # like precision
        is_aqt = not isinstance(self.quant, quantizations.Fp8Quantization)
        kw = {"mesh_axes": rhs_mesh_axes} if is_aqt else {"dtype": self.dtype}
        return self.quant.einsum(**kw)(*args)  # pytype: disable=attribute-error

      einsum_op = aqt_einsum
    else:
      einsum_op = jnp.einsum
    return einsum_op

  def maybe_all_gather_kernel_weight_in_expert_parallelism(
      self, kernel: jax.Array, kernel_axes: Tuple[Optional[str], ...]
  ):
    """All-gather kernel weight in expert parallelism if needed."""
    if self.get_expert_parallelism_size() > 1:
      # This will trigger all-gather using weight_dtype
      # relax it unless really necessary in expert parallelism only
      # Otherwise compiler will handle communication automatically
      # esp. with int8 quantization, kernel will be all-gathered in int8 instead
      # of weight_dtype
      kernel = self._maybe_shard_with_logical(kernel, kernel_axes)
    return kernel

  def dense_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
      input_ids=None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Dense matrix multiplication."""
    # gate_logits: batch, length, expert
    gate_logits = self._maybe_shard_with_logical(gate_logits, ("activation_batch_moe", "activation_length_moe", None))
    # NOTE: deepseek2 has a different pattern
    if self.config.model_name.startswith(("deepseek3", "deepseek4")):
      # pre_bias_logits is None for non-deepseek3/4 models, including deepseek2
      pre_bias_logits = self._maybe_shard_with_logical(
          pre_bias_logits, ("activation_batch_moe", "activation_length_moe", None)
      )
    top_k_weights, top_k_indices = self.get_topk(gate_logits, pre_bias_logits, self.rngs, input_ids=input_ids)
    is_llama4_decoder_layer = self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4
    if is_llama4_decoder_layer:
      router_scores = jax.nn.sigmoid(top_k_weights.astype(jnp.float32)).astype(self.dtype)
      inputs = inputs * router_scores
    else:
      weights = self.reshape_and_update_weights(top_k_weights, top_k_indices)
    matmul_precision = jax.lax.Precision(self.config.matmul_precision)

    # Calculate load balance loss
    # DeepSeek V4 uses Hash Routing for the first `first_num_hash_layers` layers.
    # These layers route deterministically based on token IDs and do not generate auxiliary loss.
    if self.config.model_call_mode != "inference" and not self.is_hash_routing:
      softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
      lb_loss = (
          self.load_balance_loss(top_k_indices, softmax_probs) if self.config.load_balance_loss_weight > 0.0 else None
      )
      # TODO(dipakg-lang, b/521990776): Add sequence-wise balance loss * 0.0001
    else:
      lb_loss = None

    # Calculate routed bias updates (loss-free)
    # The bias update logic is only applicable to Top-K routed layers.
    if self.should_update_load_balance():
      bias_updates = calculate_load_balance_updates(
          top_k_indices,
          self.config.num_experts,
          self.config.routed_bias_update_rate,
      )
    else:
      bias_updates = None

    batch_size = inputs.shape[0]
    seq_len = inputs.shape[1]

    cp, sub_seq = self.get_context_partition_and_sub_seq(seq_len)

    if self.config.capacity_factor > 0:
      # token dropping if needed
      moe_peel_expert = False  # only the training dispatch/MLP path peels 'expert' from the batch dim
      if self.config.model_call_mode != "inference":
        # TODO(b/425930949): remove this pylint by refactoring the logic here.
        dispatch_mask, combine_mask = self.generate_masks(
            top_k_indices, weights  # pylint: disable=undefined-variable,possibly-used-before-assignment
        )
        mask_axes = (
            "activation_batch_moe",
            "activation_norm_length_moe",
            None,
            None,
        )
        # Dispatch/MLP are already expert-sharded via "activation_exp". With
        # moe_dispatch_no_expert_sharding we peel 'expert' off the batch dim of these specs
        # (see _maybe_shard_moe_dispatch) so the GEMM stays expert-parallel (AllToAll) instead
        # of double-mapping E and B onto 'expert' (FSDP-style fallback).
        moe_peel_expert = self.config.moe_dispatch_no_expert_sharding
        dispatch_axis = (
            "activation_exp",
            "activation_batch_moe",
            None,
            "activation_embed_moe",
        )
        mlp_axis = (
            "activation_exp",
            "activation_batch_moe",
            None,
            "activation_mlp",
        )
        dispatch_eimsum = "BSM,BSEC -> EBCM"
        mlp_up_einsum = "EBCM,EMH -> EBCH"
        mlp_down_einsum = "EBCH,EHM -> EBCM"
        output_einsum = "EBCM,BSEC -> BSM"
      else:
        # TODO(b/425930507): Try replacing `softmax_probs` with padded weights
        # and verify with decode acc tests.
        softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        dispatch_mask, combine_mask = self.generate_masks_subgroup(top_k_indices, softmax_probs)
        if self.get_context_autoregressive_parallelism_size() > 0 and cp == 1:
          mask_axes = (
              "activation_norm_length_moe",
              "activation_batch_moe",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_norm_length_moe",
              "activation_batch_moe",
              None,
              "activation_embed_moe",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_embed_moe",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_mlp",
          )
        else:
          mask_axes = (
              "activation_batch_moe",
              "activation_norm_length_moe",
              None,
              None,
              None,
          )
          input_axis = (
              "activation_batch_moe",
              "activation_norm_length_moe",
              None,
              "activation_embed_moe",
          )
          dispatch_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_embed_moe",
          )
          mlp_axis = (
              "activation_exp",
              "activation_batch_moe",
              None,
              None,
              "activation_mlp",
          )
        dispatch_eimsum = "BNSM,BNSEC -> EBNCM"
        mlp_up_einsum = "EBNCM,EMH -> EBNCH"
        mlp_down_einsum = "EBNCH,EHM -> EBNCM"
        output_einsum = "EBNCM,BNSEC -> BNSM"

        inputs = jnp.reshape(inputs, (batch_size, cp, sub_seq, inputs.shape[2]))
        inputs = self._maybe_shard_with_logical(inputs, input_axis)

      dispatch_mask = self._maybe_shard_with_logical(dispatch_mask, mask_axes)
      combine_mask = self._maybe_shard_with_logical(combine_mask, mask_axes)

      with jax.named_scope("dispatch"):
        # only cp during prefill
        dispatch = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=DISPATCH)(
            dispatch_eimsum, inputs, dispatch_mask, precision=matmul_precision
        )
        if cp > 1:
          dispatch = self._maybe_shard_with_logical(
              dispatch,
              (
                  None,
                  "activation_batch_moe",
                  "activation_norm_length_moe",
                  None,
                  "activation_embed_moe",
              ),
          )
        dispatch = self._maybe_shard_moe_dispatch(dispatch, dispatch_axis, moe_peel_expert)
      with jax.named_scope("wi_0"):
        w0_kernel_axes = ("exp", None, "mlp")
        w0_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w0_kernel, w0_kernel_axes)
        layer_w0 = self.get_einsum(rhs_mesh_axes=w0_kernel_axes)(
            mlp_up_einsum, dispatch, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w0_bias = w0_bias[:, None, None, :]
          layer_w0 = layer_w0 + w0_bias

        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = self._maybe_shard_moe_dispatch(layer_w0, mlp_axis, moe_peel_expert)
        layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")
      with jax.named_scope("wi_1"):
        w1_kernel_axes = ("exp", None, "mlp")
        w1_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(w1_kernel, w1_kernel_axes)
        layer_w1 = self.get_einsum(rhs_mesh_axes=w1_kernel_axes)(
            mlp_up_einsum, dispatch, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          w1_bias = w1_bias[:, None, None, :]
          layer_w1 = layer_w1 + w1_bias
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = self._maybe_shard_moe_dispatch(layer_w1, mlp_axis, moe_peel_expert)
        layer_w1 = adc.checkpoint_name(adc.checkpoint_name(layer_w1, "mlpwi_1"), "moe_mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)
      with jax.named_scope("wo"):
        wo_kernel_axes = ("exp", "mlp", None)
        wo_kernel = self.maybe_all_gather_kernel_weight_in_expert_parallelism(wo_kernel, wo_kernel_axes)
        intermediate_layer = self.get_einsum(rhs_mesh_axes=wo_kernel_axes)(
            mlp_down_einsum,
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          wo_bias = wo_bias[:, None, None, :]
          intermediate_layer = intermediate_layer + wo_bias
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        if self.config.model_call_mode != "inference":
          intermediate_layer = self._maybe_shard_with_logical(
              intermediate_layer,
              (
                  "activation_exp",
                  "activation_batch_moe",
                  None,
                  "activation_embed_moe",
              ),
          )
        intermediate_layer = adc.checkpoint_name(adc.checkpoint_name(intermediate_layer, "mlpwo"), "moe_mlpwo")
      with jax.named_scope("combine"):
        # Matmul & element wise operation
        output = self.get_einsum(rhs_mesh_axes=mask_axes, einsum_name=COMBINE)(
            output_einsum,
            intermediate_layer,
            combine_mask,
            precision=matmul_precision,
        )
        if output.ndim == 4:
          output = jnp.reshape(
              output,
              (
                  output.shape[0],
                  output.shape[1] * output.shape[2],
                  output.shape[3],
              ),
          )
      return output, lb_loss, bias_updates
    else:
      inputs = self._maybe_shard_with_logical(
          inputs,
          (
              "activation_batch_moe",
              "activation_norm_length_moe",
              "activation_embed_moe",
          ),
      )
      with jax.named_scope("wi_0"):
        layer_w0 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w0_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w0 = layer_w0 + w0_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w0 = layer_w0.astype(jnp.float32)
        layer_w0 = adc.checkpoint_name(adc.checkpoint_name(layer_w0, "mlpwi_0"), "moe_mlpwi_0")
      with jax.named_scope("wi_1"):
        layer_w1 = self.get_einsum(rhs_mesh_axes=self.wi_kernel_axes)(
            "BSM,EMH -> BSEH", inputs, w1_kernel, precision=matmul_precision
        )
        if self.config.mlp_bias:
          layer_w1 = layer_w1 + w1_bias[None, None, :, :]
        if self.config.activations_in_float32:
          layer_w1 = layer_w1.astype(jnp.float32)
        layer_w1 = adc.checkpoint_name(adc.checkpoint_name(layer_w1, "mlpwi_1"), "moe_mlpwi_1")
      layer_multiply = self.apply_ffn_activation(layer_w0, layer_w1)

      with jax.named_scope("wo"):
        intermediate_layer = self.get_einsum(rhs_mesh_axes=self.wo_kernel_axes)(
            "BSEH,EHM -> BSEM",
            layer_multiply,
            wo_kernel,
            precision=matmul_precision,
        )
        if self.config.mlp_bias:
          intermediate_layer = intermediate_layer + wo_bias[None, None, :, :]
        if self.config.activations_in_float32:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
        intermediate_layer = adc.checkpoint_name(adc.checkpoint_name(intermediate_layer, "mlpwo"), "moe_mlpwo")
      with jax.named_scope("weight_sum"):
        if is_llama4_decoder_layer:
          weights = self.reshape_and_update_weights(jnp.ones_like(top_k_weights), top_k_indices)
        if self.config.float32_weight_sum:
          intermediate_layer = intermediate_layer.astype(jnp.float32)
          weights = weights.astype(jnp.float32)
        # cast to f32 for sum up in einsum op
        output = jnp.einsum(
            "BSEM,BSE -> BSM",
            intermediate_layer,
            weights,
            precision=matmul_precision,
        ).astype(self.dtype)
      return output, lb_loss, bias_updates

  def fused_moe_matmul(
      self,
      inputs,
      gate_logits,
      wo_kernel,
      w0_kernel=None,
      w1_kernel=None,
      fused_kernel=None,
  ) -> tuple[jax.Array, None, None]:
    """Fused MoE via tpu_inference fused_moe_func (vllm_rpa path only).

    fused_moe_func handles routing, GMM, and weighted combination internally.
    It does not compute lb_loss or bias_updates (inference-only).
    """
    try:
      # pylint: disable=import-outside-toplevel
      # pytype: disable=import-error
      from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
    except ImportError as e:
      raise ImportError("fused_moe_matmul requires the tpu-inference package.") from e

    # Reshape 3D [B, S, D] -> 2D [T, D] (fused_moe_func expects 2D input)
    batch_size, seq_len, emb_dim = inputs.shape
    hidden_states = jnp.reshape(inputs, (batch_size * seq_len, emb_dim))
    gating_output = jnp.reshape(gate_logits, (batch_size * seq_len, self.num_experts))

    # Concatenate gate and up projections: [E, D, H] + [E, D, H] -> [E, D, 2H]
    # fused_moe_func splits this internally: gate=w1[..., :H], up=w1[..., H:]
    if fused_kernel is None:
      fused_kernel = jnp.concatenate([w0_kernel, w1_kernel], axis=-1)

    # Use expert parallelism if the expert axis has size > 1
    use_ep = self.get_expert_parallelism_size() > 1

    # Map MaxText config fields to fused_moe_func args
    activation = self.config.mlp_activations[0]  # e.g. "silu"
    scoring_fn = self.config.routed_score_func if self.config.routed_score_func else "softmax"

    # Check if the model architecture intrinsically renormalizes weights
    renormalize = self.config.norm_topk_prob or (
        self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4)
    )

    output_2d = fused_moe_func(
        hidden_states=hidden_states,
        w1=fused_kernel,
        w2=wo_kernel,
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        gating_output=gating_output,
        topk=self.num_experts_per_tok,
        renormalize=renormalize,
        mesh=self.mesh,
        use_ep=use_ep,
        activation=activation,
        scoring_fn=scoring_fn,
    )

    # Reshape output 2D [T, D] -> 3D [B, S, D]
    output = jnp.reshape(output_2d, (batch_size, seq_len, emb_dim))
    return output, None, None

  def retrieve_quantized_weight(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[aqt.QTensor, aqt.QTensor, aqt.QTensor]:
    """Retrieve quantized weights."""
    # This is called only during tracing. This is to invoke creation of
    # quantized tensor inside AqtEinsum.  After jit, this will become no-op and
    # will not affect performance.
    _ = self.dense_matmul(
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        w0_bias,
        w1_bias,
        wo_bias,
    )

    w0_kernel = self.variables["aqt"]["AqtEinsum_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    w1_kernel = self.variables["aqt"]["AqtEinsum_1"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    wo_kernel = self.variables["aqt"]["AqtEinsum_2"]["AqtDotGeneral_0"]["qrhs"]["frozen"]

    w0_kernel = max_utils.unbox_logicallypartioned(w0_kernel)
    w1_kernel = max_utils.unbox_logicallypartioned(w1_kernel)
    wo_kernel = max_utils.unbox_logicallypartioned(wo_kernel)
    return w0_kernel, w1_kernel, wo_kernel

  def _splash_offload_sched_group(self):
    """Scheduling-group id for the combine cotangent all-gather (backward), or None.

    Returns _SPLASH_OFFLOAD_SCHED_GROUP only when BOTH moe_splash_host_offload and
    moe_splash_offload_scheduling_group are set -- so the tag exists only on the host-offload
    recovery path and every other config keeps a byte-identical schedule.
    """
    cfg = self.config
    if getattr(cfg, "moe_splash_host_offload", False) and getattr(cfg, "moe_splash_offload_scheduling_group", False):
      return _SPLASH_OFFLOAD_SCHED_GROUP
    return None

  def gather_weights(self, xlayer_w01=None, w01_only=False, wo_only=False):
    """FSDP-all-gather the routed expert weights (wi_0/wi_1/wo) early, so the
    all-gather can be emitted in the ATTENTION phase (program-order before the
    attention kernel) and overlap it.

    Returns (w0, w1, wo) gathered to the same layout sparse_matmul would use,
    for passing back as `pregathered_weights`; or None when the simple bf16
    ring path doesn't hold (prefuse / sparsity / per-expert-scale / serve-quant),
    in which case the caller falls back to the normal in-MoE gather.

    Cross-layer backward prefetch (moe_bwd_xlayer_prefetch):
      - xlayer_w01: the lifted (wi_0, wi_1) slice for THIS layer (from the Decoder-owned
        stacked param), used instead of self.wi_0/wi_1 (which are zeros placeholders when
        the lift is on). wo always lives on this module.
      - w01_only=True: gather ONLY (w0, w1) and return that 2-tuple -- the reverse-prefetch
        of the next backward layer's up-proj all-gather (also the top layer's own gather).
      - wo_only=True: gather ONLY wo (the consumer path, where w0/w1 come from swap_gather_w01).
    """
    cfg = self.config
    _fp8cv = getattr(cfg, "moe_fp8_cv_weight_ag", False)
    if not (
        (cfg.moe_weight_ag_scheduling_group or _fp8cv) and cfg.use_ring_of_experts and not cfg.shard_exp_on_fsdp
    ):
      return None
    # Only the plain path is safe to pre-gather; otherwise weights need
    # post-processing (scale/sparsity/fuse) that happens in __call__.
    if (
        cfg.prefuse_moe_weights
        or self.wi_0_sparsity_module is not None
        or self.per_expert_scale is not None
        or quantizations.in_serve_mode(self.quant)
    ):
      return None

    if xlayer_w01 is not None:
      w0 = jnp.asarray(xlayer_w01[0], self.dtype)
      w1 = jnp.asarray(xlayer_w01[1], self.dtype)
    else:
      w0 = jnp.asarray(self.wi_0[...], self.dtype)
      w1 = jnp.asarray(self.wi_1[...], self.dtype)
    wo = jnp.asarray(self.wo[...], self.dtype)
    # in = fsdp-sharded-on-embed kernel layout; out = the gathered (mlp_no_fsdp /
    # embed_tensor_transpose) layout sparse_matmul expects (default ring branch).
    wi_in = self._logical_to_mesh_axes(self.wi_kernel_axes)
    wo_in = self._logical_to_mesh_axes(self.wo_kernel_axes)
    w0_out = self._logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))
    wo_out = self._logical_to_mesh_axes(("exp", "mlp_no_fsdp", "embed_tensor_transpose"))

    # custom_vjp so the FORWARD gather carries the _scheduling_group_id (overlaps the
    # attention) while the BACKWARD/remat path re-gathers PLAINLY (no annotation)
    # and nothing big is saved. This avoids BOTH failure modes seen earlier:
    #   (1) tagging the backward gather -> the gather's reduce-scatter back-edges into
    #       the rematerialized forward -> FAILED_PRECONDITION scheduling cycle;
    #   (2) saving/offloading the full gathered weights to dodge the cycle -> ~325GB/core
    #       across the scanned layers -> HBM/host OOM.
    # The custom_vjp PRIMAL is the plain (unannotated) gather, which is what
    # remat_policy=custom recomputes in the backward -> no annotation in the rematted
    # gather -> no cycle. The custom forward rule applies the annotation (forward
    # overlap). Grad of a tiled fsdp all-gather is a tiled psum_scatter (the transpose),
    # so FSDP weight grads stay correct; nothing big is held as a residual.
    def _make_cv_gather(in_pspec, out_pspec, gather_axis, sched_group):
      @jax.custom_vjp
      def _g(w):  # PRIMAL: plain gather (what remat recomputes in the backward)
        return jax.shard_map(
            lambda x: jax.lax.all_gather(x, "fsdp", axis=gather_axis, tiled=True),
            mesh=self.mesh, in_specs=(in_pspec,), out_specs=out_pspec, check_vma=False)(w)

      def _g_fwd(w):  # FORWARD under diff: annotated gather (overlaps attention)
        def _fn(x):
          with _scheduling_group(sched_group):
            return jax.lax.all_gather(x, "fsdp", axis=gather_axis, tiled=True)
        w_full = jax.shard_map(_fn, mesh=self.mesh, in_specs=(in_pspec,), out_specs=out_pspec, check_vma=False)(w)
        return w_full, None  # no big residual saved (sharded w is recomputed cheaply / not needed)

      def _g_bwd(_res, ct):  # transpose of tiled all-gather over fsdp = tiled psum_scatter
        g_sharded = jax.shard_map(
            lambda gg: jax.lax.psum_scatter(gg, "fsdp", scatter_dimension=gather_axis, tiled=True),
            mesh=self.mesh, in_specs=(out_pspec,), out_specs=in_pspec, check_vma=False)(ct)
        return (g_sharded,)

      _g.defvjp(_g_fwd, _g_bwd)
      return _g

    def _cv_scale(w):
      # DYNAMIC per-output-channel scale on the GLOBAL param: max over (exp, k) -> [1,1,n]. The max
      # over the fsdp-sharded dim is a tiny GSPMD all-reduce ([1,1,n], negligible -- and for wo it
      # crosses no shard at all). stop_gradient: forward-only scale (STE backward).
      # checkpoint_name: SAVE the [1,1,n] scale (tiny) under remat_policy=custom so the rematted
      # backward (which re-runs quantize+gather for the e4m3 bwd re-gather) LOADS it instead of
      # re-running the cross-shard max reduction inside the checkpoint scope. Config key
      # moe_fp8_scale defaults to 'device'.
      sc = jax.lax.stop_gradient(
          jnp.max(jnp.abs(w), axis=(0, 1), keepdims=True).astype(jnp.float32) / 448.0 + 1e-20)
      return adc.checkpoint_name(sc, "moe_fp8_scale")

    # Distinct scheduling-group ids per weight so the all-gather-combiner cannot
    # fuse the three into one un-hideable monolith; each smaller gather can
    # then be scheduled independently behind different attention-phase compute.
    # NO optimization_barrier: it is self-dual, so a barrier on the gathered weight
    # fences the weight-grad feeding the backward psum_scatter -> pins the RS exposed.
    # The distinct group ids already prevent the all-gather-combiner fusion.
    if _fp8cv and not w01_only and not wo_only:
      # fp8 path (plain 3-tuple only; the handwritten/xlayer variants stay bf16): quantize the
      # STORAGE-SHARDED param to e4m3 via _ste_quant (STE custom_vjp: bf16 ct pass-through -- the
      # sharded ct arriving here IS the final scattered weight grad, no rescale/reduce needed) and
      # return the SHARDED qvalue. The e4m3 all-gather happens INSIDE the sparse_matmul body (its
      # autodiff transpose = ONE direct psum_scatter to storage sharding -- the efficient single-RS
      # weight-grad form; the earlier mini-shard_map gather here left the main boundary replicated,
      # whose transpose AR + reshard failed to fuse into an RS: two slow RSs @ 22-38 GB/s, +0.35s).
      # NOTE moe_fp8_cv_weight_ag_tags is DEPRECATED/ignored: the tags variant NaN'd on cluster
      # round 2 (cvwag2t) and was dominated by no-tags (5.302 vs 5.106); with the in-body gather
      # there is no pre-attention gather to tag. Do not re-enable without a fresh numerics gate.
      sc0, sc1, sco = _cv_scale(w0), _cv_scale(w1), _cv_scale(wo)
      w0 = qpl.QArray(qvalue=_ste_quant(w0, sc0), scale=sc0)
      w1 = qpl.QArray(qvalue=_ste_quant(w1, sc1), scale=sc1)
      wo = qpl.QArray(qvalue=_ste_quant(wo, sco), scale=sco)
      return (w0, w1, wo)
    if w01_only:
      # Reverse-prefetch gather of the NEXT backward layer's w0/w1 only (no wo). The producer caller
      # stop_gradients the result (pure scheduling: the consuming layer routes the grad via swap_gather's
      # psum_scatter), and the top backward layer uses THIS as its own (grad-live) gather. Returns (w0,w1).
      w0 = _make_cv_gather(wi_in, w0_out, 1, _WEIGHT_AG_SCHED_GROUP)(w0)
      w1 = _make_cv_gather(wi_in, w0_out, 1, _WEIGHT_AG_SCHED_GROUP + 1)(w1)
      return (w0, w1)
    if wo_only:
      # Reverse-prefetch consumer: w0/w1 come from swap_gather_w01 (the early-emitted handed all-gather);
      # gather ONLY wo here (grad -> self.wo). Returns the gathered wo tensor.
      return _make_cv_gather(wo_in, wo_out, 2, _WEIGHT_AG_SCHED_GROUP + 2)(wo)
    w0 = _make_cv_gather(wi_in, w0_out, 1, _WEIGHT_AG_SCHED_GROUP)(w0)
    w1 = _make_cv_gather(wi_in, w0_out, 1, _WEIGHT_AG_SCHED_GROUP + 1)(w1)
    wo = _make_cv_gather(wo_in, wo_out, 2, _WEIGHT_AG_SCHED_GROUP + 2)(wo)
    return (w0, w1, wo)

  def swap_gather_w01(self, handed_w0, handed_w1, xlayer_w01):
    """Reverse-prefetch CONSUMER for w0/w1 (moe_bwd_xlayer_prefetch).

    Returns (w0, w1) whose VALUE is the handed, already-gathered up-proj weights (all-gathered
    ONE backward-layer early by the producer and handed down the reverse scan carry, so THIS
    layer's own up-proj all-gather is not emitted), and whose BACKWARD is the tiled fsdp
    psum_scatter (transpose of the all-gather) routed to `xlayer_w01` -- i.e. d(w0/w1) flows to
    the FSDP-sharded lifted slice exactly as the plain gather's psum_scatter would. The handed
    value gets ZERO cotangent (its grad path is here, not at the producer). Value == the plain
    all-gather of xlayer_w01 (bit-identical), and the grad == the plain gather's psum_scatter, so
    this is bit-exact vs the non-prefetch path -- it only moves WHERE the all-gather is emitted.
    """
    wi_in = self._logical_to_mesh_axes(self.wi_kernel_axes)
    w0_out = self._logical_to_mesh_axes(("exp", "embed_tensor_transpose", "mlp_no_fsdp"))

    def _make_swap():
      @jax.custom_vjp
      def _s(handed, loc):  # value = handed (already gathered); loc only pins the grad target
        return handed

      def _s_fwd(handed, loc):
        return handed, None

      def _s_bwd(_res, ct):  # transpose of tiled fsdp all-gather = tiled psum_scatter -> sharded loc grad
        g_loc = jax.shard_map(
            lambda gg: jax.lax.psum_scatter(gg, "fsdp", scatter_dimension=1, tiled=True),
            mesh=self.mesh, in_specs=(w0_out,), out_specs=wi_in, check_vma=False)(ct)
        return (jnp.zeros_like(ct), g_loc)  # zero grad to handed; real grad to the sharded slice

      _s.defvjp(_s_fwd, _s_bwd)
      return _s

    w0 = _make_swap()(jnp.asarray(handed_w0, self.dtype), jnp.asarray(xlayer_w01[0], self.dtype))
    w1 = _make_swap()(jnp.asarray(handed_w1, self.dtype), jnp.asarray(xlayer_w01[1], self.dtype))
    return (w0, w1)

  def __call__(
      self,
      inputs: jax.Array,
      input_ids: jax.Array | None = None,
      gate_inputs: jax.Array | None = None,
      out_sharding: NamedSharding | None = None,
      pregathered_weights: tuple | None = None,
      use_chunked_combine: bool = True,
      use_chunked_dispatch: bool = True,
      return_combine_token: bool = False,
      save_routing: bool = False,
      saved_routing=None,
      bwd_direct_token_ag: bool = False,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Executes the routed MoE block.

    `pregathered_weights`, if given, are (w0, w1, wo) already FSDP-all-gathered
    by `gather_weights` in the attention phase; they replace the in-block read
    (the boundary gather at the shard_map entry becomes a no-op, so there is no
    double gather). Only used on the plain bf16 ring path.

    Args:
      inputs: The input activations.
      input_ids: Optional token IDs corresponding to the inputs, used strictly for Hash Routing
        in DeepSeek V4's early MoE layers. If None, routing relies purely on `gate_inputs` or `inputs`.
      gate_inputs: Optional alternate inputs to feed into the routing gate.
      out_sharding: Optional sharding specification for the output.

    Returns:
      A tuple containing the MoE output, the load balance loss (if applicable),
      and any routed bias updates.
    """
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_dtype = jnp.float32 if cfg.float32_gate_logits else cfg.dtype
    routing_inputs = inputs if gate_inputs is None else gate_inputs.astype(gate_dtype)
    gate_logits, pre_bias_logits = self.gate(routing_inputs)

    fused_kernel = None
    w0_kernel = None
    w1_kernel = None
    if pregathered_weights is not None:
      # Already FSDP-gathered in the attention phase; skip the in-block read
      # (gather_weights bailed out of every path that needs post-processing).
      w0_kernel, w1_kernel, wo_kernel = pregathered_weights
    else:
      wo_kernel = jnp.asarray(self.wo[...], self.dtype)

      if cfg.prefuse_moe_weights and cfg.attention in ("vllm_rpa", "vllm_batched_rpa") and not self.is_hash_routing:
        fused_kernel = jnp.asarray(self.wi[...], self.dtype)
      elif cfg.prefuse_moe_weights:
        wi = jnp.asarray(self.wi[...], self.dtype)
        n = wi.shape[-1] // 2
        w0_kernel = wi[..., :n]
        w1_kernel = wi[..., n:]
      else:
        w0_kernel = jnp.asarray(self.wi_0[...], self.dtype)
        w1_kernel = jnp.asarray(self.wi_1[...], self.dtype)

    # Only apply per expert scales if we have not fused with the out-projections at init time.
    if self.per_expert_scale is not None and cfg.model_call_mode != "inference" and not cfg.fuse_expert_scales:
      wo_kernel = wo_kernel * jnp.asarray(self.per_expert_scale[...], self.dtype)[:, None, None]

    if self.wi_0_sparsity_module is not None:
      _, w0_kernel = self.wi_0_sparsity_module(jnp.zeros_like(w0_kernel), w0_kernel)
      _, w1_kernel = self.wi_1_sparsity_module(jnp.zeros_like(w1_kernel), w1_kernel)
      _, wo_kernel = self.wo_sparsity_module(jnp.zeros_like(wo_kernel), wo_kernel)
    if cfg.mlp_bias:
      w0_bias = jnp.asarray(self.wi_0_bias[...], self.dtype)
      w1_bias = jnp.asarray(self.wi_1_bias[...], self.dtype)
      wo_bias = jnp.asarray(self.wo_bias[...], self.dtype)
    else:
      w0_bias, w1_bias, wo_bias = None, None, None

    # vllm_rpa codepath uses fused_moe_func from tpu_inference for optimized inference.
    # The fused MoE kernel currently only supports standard Top-K routing with associated
    # weights. Hash routed layers bypass this kernel and fall back
    # to the sparse matmul implementation.
    if (save_routing or saved_routing is not None) and not (cfg.attention != "vllm_rpa" and cfg.sparse_matmul):
      raise ValueError("moe_save_sort_indices requires the sparse_matmul path (non-vllm_rpa).")
    if cfg.attention in ("vllm_rpa", "vllm_batched_rpa") and not self.is_hash_routing:
      output, lb_loss, bias_updates = self.fused_moe_matmul(
          inputs,
          gate_logits,
          wo_kernel,
          w0_kernel=w0_kernel,
          w1_kernel=w1_kernel,
          fused_kernel=fused_kernel,
      )
    elif cfg.sparse_matmul:
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs,
            gate_logits,
            pre_bias_logits,
            w0_kernel,
            w1_kernel,
            wo_kernel,
            w0_bias,
            w1_bias,
            wo_bias,
        )
      result = self.sparse_matmul(
          inputs,
          gate_logits,
          pre_bias_logits,
          w0_kernel,
          w1_kernel,
          wo_kernel,
          w0_bias,
          w1_bias,
          wo_bias,
          input_ids,
          use_chunked_combine=use_chunked_combine,
          use_chunked_dispatch=use_chunked_dispatch,
          return_combine_token=return_combine_token,
          save_routing=save_routing,
          saved_routing=saved_routing,
          bwd_direct_token_ag=bwd_direct_token_ag,
      )
      # 3-tuple, +combine scheduling token (possibly None) when return_combine_token=True
      # (moe_shared_after_combine), +the per-chunk routing bundle LAST when save_routing=True
      # (moe_save_sort_indices).
      return result
    else:
      output, lb_loss, bias_updates = self.dense_matmul(
          inputs,
          gate_logits,
          pre_bias_logits,
          w0_kernel,
          w1_kernel,
          wo_kernel,
          w0_bias,
          w1_bias,
          wo_bias,
          input_ids,
      )
    if return_combine_token:
      return output, lb_loss, bias_updates, None
    return output, lb_loss, bias_updates


class RoutedAndSharedMoE(nnx.Module):
  """Implements a block which combines shared and routed experts."""

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      kernel_init: NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
      is_hash_routing: bool = False,
  ):
    """Initializes the RoutedAndSharedMoE module.

    Attributes:
      config: The main config setting.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
      is_hash_routing: Passed down to the internal `RoutedMoE` to determine routing behavior (e.g. hash vs top-K).
    """
    self.config = config
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    self.is_hash_routing = is_hash_routing
    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    # NOTE: the name MoeBlock_0 is to ensure reverse compatibility with
    # existing checkpoints for routed experts.
    self.MoeBlock_0 = RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_moe", None),
        intermediate_dim=self.config.moe_mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
        is_hash_routing=self.is_hash_routing,
    )

    shared_expert_mlp_dim = maxtext_utils.get_shared_expert_mlp_dim(self.config)
    self.shared_experts = linears.MlpBlock(
        mesh=self.mesh,
        in_features=self.moe_expert_input_dim,
        intermediate_dim=self.config.shared_experts * shared_expert_mlp_dim,
        activations=self.config.mlp_activations,
        kernel_init=self.kernel_init,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=self.quant,
        rngs=self.rngs,
    )

  @property
  def routed_moe(self):
    return self.MoeBlock_0

  def gather_routed_weights(self, xlayer_w01=None, w01_only=False, wo_only=False):
    """Pre-gather the routed experts' FSDP weights (see RoutedMoE.gather_weights).
    Call this in the attention phase; pass the result back as pregathered_weights.
    xlayer_w01 / w01_only / wo_only: cross-layer backward prefetch (moe_bwd_xlayer_prefetch)."""
    return self.MoeBlock_0.gather_weights(xlayer_w01=xlayer_w01, w01_only=w01_only, wo_only=wo_only)

  def swap_gather_routed_w01(self, handed_w0, handed_w1, xlayer_w01):
    """Reverse-prefetch consumer for w0/w1 (see RoutedMoE.swap_gather_w01)."""
    return self.MoeBlock_0.swap_gather_w01(handed_w0, handed_w1, xlayer_w01)

  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      gate_inputs: jax.Array | None = None,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
      input_ids: jax.Array | None = None,
      pregathered_weights: tuple | None = None,
      use_chunked_combine: bool = True,
      use_chunked_dispatch: bool = True,
      save_routing: bool = False,
      saved_routing=None,
      bwd_direct_token_ag: bool = False,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Executes both the routed experts and the shared expert block.

    Args:
      inputs: The input activations.
      original_inputs: The original pre-normalization inputs (unused by default).
      gate_inputs: Optional alternate inputs to feed into the routing gate.
      intermediate_sharding: Optional sharding spec for the shared experts intermediate output.
      out_sharding: Optional sharding spec for the final combined output.
      input_ids: Optional token IDs corresponding to the inputs, used strictly for Hash Routing
        in DeepSeek V4's early MoE layers.

    Returns:
      A tuple containing the combined MoE output (routed + shared),
      the load balance loss, and any routed bias updates.
    """
    want_token = self.config.moe_shared_after_combine
    result = self.routed_moe(
        inputs,
        gate_inputs=gate_inputs,
        out_sharding=out_sharding,
        input_ids=input_ids,
        pregathered_weights=pregathered_weights,
        use_chunked_combine=use_chunked_combine,
        use_chunked_dispatch=use_chunked_dispatch,
        return_combine_token=want_token,
        save_routing=save_routing,
        saved_routing=saved_routing,
        bwd_direct_token_ag=bwd_direct_token_ag,
    )
    # Unpack: (out, lb, bias) [+ combine_token if want_token] [+ routing bundle if save_routing].
    routing_saved = None
    if save_routing:
      result, routing_saved = result[:-1], result[-1]
    if want_token:
      routed_experts, load_balance_loss, moe_bias_updates, combine_token = result
    else:
      routed_experts, load_balance_loss, moe_bias_updates = result
      combine_token = None
    shared_input = inputs
    if combine_token is not None:
      # moe_shared_after_combine DEADLINE FENCE: tie the shared-expert MLP's input to the
      # routed path's FIRST-chunk pre-RS combined output. The shared expert (dense TC GMM on
      # every token, data-independent of the routed combine) is otherwise scheduled EARLY,
      # leaving the chunk reduce-scatters exposed at layer-end with the TC idle; this fence
      # forbids scheduling it before the combine phase begins, pushing it into the chunk-RS
      # window. Deliberately NOT fenced on any RS output or on the routed output (either
      # would serialize the pipeline). Identity on values -> bit-exact. Same caveat as the
      # chunk barriers: xla_tpu_aggressive_opt_barrier_removal=true may strip this fence.
      shared_input, _ = jax.lax.optimization_barrier((inputs, combine_token))
    shared_experts = self.shared_experts(
        shared_input, intermediate_sharding=intermediate_sharding, out_sharding=out_sharding
    )
    if save_routing:
      return routed_experts + shared_experts, load_balance_loss, moe_bias_updates, routing_saved
    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


def get_gate_logit(
    inputs_shape: tuple[int, ...],
    out_features_shape: Union[Iterable[int], int],
    model_name: str,
    axis: Union[Iterable[int], int] = -1,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: Tuple[Optional[str], ...] = (),
    use_bias: bool = False,
    score_func: str = "",
    quant: Optional[quantizations.AqtQuantization] = None,
    matmul_precision: str = "default",
    name: Optional[str] = None,
):
  """Creates a GateLogit Linen module."""

  axis = linears.canonicalize_tuple(axis)
  in_features_shape = tuple(inputs_shape[ax] for ax in linears.normalize_axes(axis, len(inputs_shape)))

  module = nnx_wrappers.to_linen(
      GateLogit,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      model_name=model_name,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      use_bias=use_bias,
      score_func=score_func,
      quant=quant,
      matmul_precision=matmul_precision,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


def get_routed_moe(
    config: ctypes.Config,
    num_experts: int,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    intermediate_dim: int = 2048,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  """Creates a RoutedMoE Linen module."""

  module = nnx_wrappers.to_linen(
      RoutedMoE,
      config=config,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      intermediate_dim=intermediate_dim,
      weight_dtype=weight_dtype,
      dtype=dtype,
      quant=quant,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


def get_routed_and_shared_moe(
    config: ctypes.Config,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
    is_hash_routing: bool = False,
):
  """Creates a RoutedAndSharedMoE Linen module."""

  module = nnx_wrappers.to_linen(
      RoutedAndSharedMoE,
      config=config,
      mesh=mesh,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      weight_dtype=weight_dtype,
      dtype=dtype,
      quant=quant,
      name=name,
      is_hash_routing=is_hash_routing,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
