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

"""Qwen3.5 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any, cast

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN, Array, Config, ShardMode, get_weight_dtype
from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers.moe import barrier_tie
from maxtext.layers.normalizations import Qwen3NextRMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.qwen3 import (
    Qwen3NextFullAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import (
    create_sharding,
    dense_kernel_gathered_sharding,
    get_logical_axis_rules,
    maybe_shard_with_logical,
    maybe_shard_with_name,
)

# -----------------------------------------
# Qwen3.5 Layer Implementations
# -----------------------------------------


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
  """Qwen3.5 GatedDeltaNet layer that is identical to Qwen3-Next GatedDeltaNet"""


class Qwen3_5FullAttention(Qwen3NextFullAttention):
  """Qwen3.5 Gated Attention layer that is identical to Qwen3-Next"""


class Qwen3_5SparseMoEBlock(Qwen3NextSparseMoeBlock):
  """Shares same MoE code as Qwen3-Next"""


class Qwen3_5ScannableBlock(nnx.Module):
  """Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers."""

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    cfg = self.config

    # Explicitly instantiate Qwen3_5DecoderLayer here
    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer_rngs = self.rngs.fork()
      layer_name = f"layer_{i}"
      layer = Qwen3_5DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          layer_idx=i,
          rngs=layer_rngs,
      )
      setattr(self, layer_name, layer)

  def _scan_weight_prefetch_modules(self) -> list[tuple[int, str, nnx.Module]]:
    """The dense-kernel modules `scan_weight_prefetch` gathers, as `(sub-layer index, path, module)`.

    `attn_q_shared_wi`: the full-attention sub-layer's query kernel and every sub-layer's shared-expert
    wi kernels (the two gathers measured exposed inside the KV-gather window). `attn_layer`: the
    full-attention sub-layer's query, key, value and output kernels, its shared-expert wi/wo kernels
    and its router gate kernel.
    """
    cfg = self.config
    layers = [getattr(self, f"layer_{i}") for i in range(cfg.inhomogeneous_layer_cycle_interval)]

    def shared_wi_wo(layer, i):
      mlp = layer.mlp.shared_expert
      if mlp is None:
        return [], []
      if cfg.fused_mlp:
        wi = [(i, f"layer_{i}.mlp.shared_expert.wi", mlp.wi)]
      else:
        names = ["wi"] if len(mlp.activations) == 1 else [f"wi_{k}" for k in range(len(mlp.activations))]
        wi = [(i, f"layer_{i}.mlp.shared_expert.{n}", getattr(mlp, n)) for n in names]
      return wi, [(i, f"layer_{i}.mlp.shared_expert.wo", mlp.wo)]

    def attention_projections(layer, i, all_of_them):
      attn = layer.attention.attention
      prefix = f"layer_{i}.attention.attention"
      if cfg.fused_qkv:
        mods = [(i, f"{prefix}.qkv_proj", attn.qkv_proj)]
      else:
        mods = [(i, f"{prefix}.query", attn.query)]
        if all_of_them:
          mods.append((i, f"{prefix}.key", attn.key))
          if not attn.share_kv_projections:
            mods.append((i, f"{prefix}.value", attn.value))
      if all_of_them:
        mods.append((i, f"{prefix}.out", attn.out))
      return mods

    selected = []
    if cfg.scan_weight_prefetch_set == "attn_q_shared_wi":
      for i, layer in enumerate(layers):
        if isinstance(layer.attention, Qwen3_5FullAttention):
          selected += attention_projections(layer, i, all_of_them=False)
      for i, layer in enumerate(layers):
        selected += shared_wi_wo(layer, i)[0]
    elif cfg.scan_weight_prefetch_set == "attn_layer":
      for i, layer in enumerate(layers):
        if not isinstance(layer.attention, Qwen3_5FullAttention):
          continue
        selected += attention_projections(layer, i, all_of_them=True)
        wi, wo = shared_wi_wo(layer, i)
        selected += wi + wo
        selected.append((i, f"layer_{i}.mlp.routed_experts.gate", layer.mlp.routed_experts.gate))
    else:
      raise ValueError(f"unknown scan_weight_prefetch_set {cfg.scan_weight_prefetch_set!r}")
    return selected

  def _prefetch_dense_kernels(self, modules):
    """Explicit all-gathers of the selected kernels, in each module's compute dtype.

    Each kernel is constrained to its physical sharding with the fsdp, fsdp_transpose, context and
    expert axes peeled (`dense_kernel_gathered_sharding`), which is the gather the dot partitioner
    would insert on its own; once the value is already in that layout the partitioner inserts no
    second gather. The cast mirrors `DenseGeneral.__call__` / `GateLogit.__call__`, which cast the
    kernel they read to their `dtype` before the dot, so the gathered bytes are the same as today's.
    With `scan_weight_prefetch_sparse_core_id >= 0` the gathers run under `compute_on("tpu_sparsecore")`
    pinned to that core, as `moe_pin_sparse_core_all_gathers` does for the expert weights.
    """
    cfg = self.config
    rules = get_logical_axis_rules()
    core_id = cfg.scan_weight_prefetch_sparse_core_id

    def gather(kernel, sharding):
      return maybe_shard_with_name(kernel, sharding, cfg.shard_mode, debug_sharding=cfg.debug_sharding)

    if core_id >= 0:
      gather = compute_on(
          gather,
          compute_type="tpu_sparsecore",
          out_memory_spaces=jax.memory.Space.Device,
          compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
      )

    gathered = []
    for _, _, mod in modules:
      kernel = jnp.asarray(mod.kernel[...], mod.dtype)
      gathered.append(gather(kernel, dense_kernel_gathered_sharding(mod.kernel_axes, self.mesh, rules)))
    return gathered

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | tuple[Any, ...] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ) -> tuple[Array, None | tuple[Any, ...]]:
    cfg = self.config
    x = carry
    # Inference with externally-managed (vLLM) caches hands this block one
    # entry per sub-layer of the cycle; the decoder groups the flat per-layer
    # list before the scan and ungroups the returned tuple afterwards.
    updated_kv_caches = []

    # scan_weight_prefetch=block_tie (design note scan_prefetch_design_1014.md, step 1): gather the selected dense
    # kernels at the top of the block and tie them to the input of sub-layer `tie_sublayer` with a forward-only
    # optimization barrier. Nothing computed from the tied `x` can start before the gathers are done, so the
    # scheduler has to issue them under sub-layers 0..tie_sublayer-1 instead of inside the attention sub-layer,
    # where today they queue behind the context-parallel KV sequence gathers on the SparseCore. The barrier is the
    # identity and the gathered bytes are the ones the partitioner would have gathered anyway, so the math is
    # unchanged; the backward is untouched (it regathers as before). Kernels consumed by a sub-layer before the tie
    # point cannot be ordered by it (their own dot already forces the gather) and are left to the partitioner.
    tie_sublayer = -1
    tied: list[tuple[int, str, nnx.Module]] = []
    tied_kernels: list[Array] = []
    # getattr: the block is also driven with stub configs that carry only the cycle interval (nnx_decoders_test).
    if getattr(cfg, "scan_weight_prefetch", None) == "block_tie" and kv_cache is None:
      tie_sublayer = cfg.scan_weight_prefetch_tie_sublayer
      tied = [m for m in self._scan_weight_prefetch_modules() if m[0] >= tie_sublayer]
      if tied:
        tied_kernels = self._prefetch_dense_kernels(tied)
        max_logging.info(
            f"scan_weight_prefetch=block_tie: {len(tied_kernels)} dense kernel gather(s) tied to the input of"
            f" sub-layer {tie_sublayer}: {', '.join(path for _, path, _ in tied)}"
        )

    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      if i == tie_sublayer and tied_kernels:
        # Both outputs of the tie replace both inputs downstream: `x` feeds this and the later sub-layers, and the
        # gathered kernels are installed into their modules, whose dots read them.
        x, tied_kernels = barrier_tie(x, tuple(tied_kernels))
        for (_, _, mod), kernel in zip(tied, tied_kernels):
          mod.kernel.value = kernel
      # forced_routed_experts, when present, is shaped
      # [inhomogeneous_layer_cycle_interval, batch, seq, top_k]: one slice per
      # sub-layer in this cycle (see nnx_decoders.py's scan wiring).
      layer_forced_routed_experts = forced_routed_experts[i] if forced_routed_experts is not None else None
      layer_kv_cache = kv_cache[i] if (kv_cache is not None and i < len(kv_cache)) else None
      x, new_kv_cache = layer(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
          kv_cache=layer_kv_cache,
          attention_metadata=attention_metadata,
          forced_routed_experts=layer_forced_routed_experts,
      )
      updated_kv_caches.append(new_kv_cache)

    if kv_cache is None:
      return x, None
    return x, tuple(updated_kv_caches)


class Qwen3_5DecoderLayer(nnx.Module):
  """
  This layer is a hybrid, capable of functioning as either:
  1. A standard attention + MoE layer.
  2. A linear attention + MoE layer.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_activation_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_mlp",
    )

    # Physical shardings used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO the callees ignore these and let GSPMD infer the layout.
    if cfg.shard_mode == ShardMode.EXPLICIT:
      self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
      self.mlp_intermediate_sharding = create_sharding(
          mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules()
      )
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
          extra_stack_level=1,
      )
    else:
      self.out_sharding = None
      self.mlp_intermediate_sharding = None
      self._maybe_shard_with_logical = lambda inputs, *args, **kwargs: inputs

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "norm"),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer.
    is_full_attention_layer = (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0

    # Conditionally instantiate either the Linear Attention or Full Attention block.
    if is_full_attention_layer:
      self.attention = Qwen3_5FullAttention(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          layer_idx=self.layer_idx,
          rngs=rngs,
      )
    else:
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
      self.attention = Qwen3_5GatedDeltaNet(
          config=cfg,
          inputs_shape=dummy_inputs_shape,
          mesh=self.mesh,
          dtype=cfg.dtype,
          model_mode=model_mode,
          quant=self.quant,
          rngs=rngs,
      )

    # Second LayerNorm, applied before the MoE block (outputs fp32 into MoE router when float32_gate_logits=True).
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=jnp.float32 if cfg.float32_gate_logits else cfg.dtype,
        weight_dtype=jnp.float32 if cfg.float32_gate_logits else get_weight_dtype(cfg, "norm"),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Instantiate our `Qwen3_5SparseMoEBlock`.
    self.mlp = Qwen3_5SparseMoEBlock(config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs)

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    residual = inputs

    if isinstance(attention_metadata, dict):
      attention_metadata = attention_metadata.get(f"layer.{self.layer_idx}", attention_metadata.get(self.layer_idx))

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3_5FullAttention):
      attention_output, new_kv_cache = cast(Qwen3_5FullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )
    else:
      attention_output, new_kv_cache = cast(Qwen3_5GatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )

    # First residual connection after attention
    attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
    hidden_states = residual + attention_output
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Prepare for the MoE block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE block.
    hidden_states = self.post_attention_layernorm(hidden_states, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Padding tokens (segment id 0) are excluded from MoE routing when moe_exclude_padding_from_routing.
    token_mask = None
    if (
        self.config.moe_exclude_padding_from_routing
        and decoder_segment_ids is not None
        and model_mode == MODEL_MODE_TRAIN
    ):
      token_mask = decoder_segment_ids != 0

    # Instantiate and call our `Qwen3_5SparseMoEBlock`.
    mlp_output, load_balance_loss = self.mlp(
        hidden_states,
        deterministic=deterministic,
        forced_routed_experts=forced_routed_experts,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
        token_mask=token_mask,
    )

    # We sow the load balancing loss so it can be collected and added to the total loss
    # during training.
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    # Final residual connection (after the MoE block)
    mlp_output = self._maybe_shard_with_logical(mlp_output, self.activation_axis_names)
    layer_output = residual + mlp_output
    layer_output = self._maybe_shard_with_logical(
        layer_output,
        self.activation_axis_names,
    )
    return layer_output, new_kv_cache


Qwen3_5DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3_5DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)


Qwen3_5ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3_5ScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
