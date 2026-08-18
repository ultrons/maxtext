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

"""Linear Layers."""

import functools
import operator
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from jax import lax
from jax.sharding import NamedSharding, Mesh, PartitionSpec
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import xla_metadata

from flax import nnx
import flax.linen as nn

from maxtext.common.common_types import DecoderBlockType, ShardMode, DType, Array, Config
from maxtext.common.common_types import MODEL_MODE_PREFILL
from maxtext.layers import nnx_wrappers, quantizations
from maxtext.layers import normalizations
from maxtext.layers.initializers import NdInitializer, nd_dense_init, default_bias_init, variable_to_logically_partitioned
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical
from maxtext.utils.sharding import maybe_shard_with_name
from maxtext.utils.sharding import get_physical_spec_without_axes
from maxtext.utils.sharding import FSDP_MESH_AXES
from maxtext.utils.sharding import truncate_out_sharding

import os

# Step-0 remat-survival probe for the start/done pattern (see kernels/startdone.py).
_STARTDONE_PROBE = os.environ.get("STARTDONE_PROBE", "0") == "1"


def _convert_to_activation_function(fn_or_string: str | Callable[..., Any]) -> Callable[..., Any]:
  """Convert a string to an activation function."""
  if fn_or_string == "linear":
    return lambda x: x
  elif fn_or_string == "sqrtsoftplus":
    # Custom activation function used by DeepSeek V4 Top-K MoE router
    return lambda x: jnp.sqrt(jax.nn.softplus(x))
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError(
        f"""Don't know how to convert {fn_or_string}
                         to an activation function"""
    )


def normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _compute_dot_general(inputs, kernel, kernel_axes, axis, contract_ind, matmul_precision, quant):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant:
    dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
    dot_general = dot_general_cls()
    return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
  return dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision)


def _compute_dot_general_nnx(
    inputs,
    kernel,
    axis,
    contract_ind,
    matmul_precision,
    quant_dot_general: nnx_wrappers.ToNNX | None,
    initializing: bool,
    out_sharding: NamedSharding | None = None,
):
  """Computes a dot_general operation that may be quantized."""
  dot_general = lax.dot_general
  matmul_precision = lax.Precision(matmul_precision)
  if quant_dot_general is not None:
    if initializing:
      quant_dot_general.lazy_init(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None)
    return quant_dot_general(inputs, kernel, ((axis, contract_ind), ((), ())), precision=None, mutable=["aqt"])

  if out_sharding is not None:
    out_ndim = (inputs.ndim - len(axis)) + (kernel.ndim - len(contract_ind))
    out_sharding = truncate_out_sharding(out_sharding, out_ndim)

  return dot_general(
      inputs, kernel, ((axis, contract_ind), ((), ())), precision=matmul_precision, out_sharding=out_sharding
  )


class DenseGeneral(nnx.Module):
  """A linear transformation with flexible axes."""

  def __init__(
      self,
      in_features_shape: Iterable[int] | int,
      out_features_shape: Iterable[int] | int,
      axis: Iterable[int] | int = -1,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = (),
      quant: None | Quant = None,
      use_bias: bool = False,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      mesh: Mesh | None = None,
      use_two_stage_all_gather: bool = False,
      debug_sharding: bool = False,
      hoist_weight_ag_sched_group: int = -1,
      *,  # Following arguments are keyword-only
      rngs: nnx.Rngs = None,
  ):
    """Initializes the DenseGeneral module.

    Args:
      in_features_shape: tuple with numbers of input features for axes specified in
        'axis'.
      out_features_shape: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      quant: quantization config, defaults to None implying no quantization.
      use_bias: whether to add bias in linear transformation.
      shard_mode: auto or explicit shard mode.
      matmul_precision: Precision for matrix multiplication.
      parameter_memory_host_offload: Determines whether to offload params to host
      mesh: Mesh of devices and physical axes, needed for two-stage all-gather.
      use_two_stage_all_gather: when the kernel is sharded on both the fsdp and
        fsdp_transpose axes, gather the two axes with two separate all-gather
        calls (separated by an optimization barrier) to avoid the relayout
        transpose XLA emits for a single combined 2-axis all-gather.
      debug_sharding: when True, log the logical/physical sharding of the
        two-stage all-gather constraints to the sharding dump files.
      hoist_weight_ag_sched_group: when >= 0, issue the FSDP weight all-gather
        EXPLICITLY (a sharding constraint that drops the fsdp axes) before the
        dot, tagged with this XLA `_scheduling_group_id`, instead of leaving it
        to SPMD to insert next to the matmul. -1 leaves the default behaviour.
      rngs: RNG state for initialization in nnx.
    """
    self.in_features_shape = canonicalize_tuple(in_features_shape)
    self.out_features_shape = canonicalize_tuple(out_features_shape)
    self.axis = canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.quant = quant
    self.use_bias = use_bias
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.mesh = mesh
    self.use_two_stage_all_gather = use_two_stage_all_gather
    self.hoist_weight_ag_sched_group = hoist_weight_ag_sched_group
    self.debug_sharding = debug_sharding

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
          sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          sharding=bias_axes,
      )
    else:
      self.bias = None

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      block_size = getattr(quant, "get_block_size", lambda: 1)()  # needed for TE MXFP8
      dummy_inputs = jnp.zeros((block_size, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None

  @property
  def quant_dot_general(self) -> nnx_wrappers.ToNNX | None:
    if self._quant_dot_general_name is None:
      return None
    return getattr(self, self._quant_dot_general_name)

  def _maybe_two_stage_all_gather(self, kernel):
    """Gather a 2D-FSDP-sharded MLP kernel with two single-axis all-gathers.

    When the kernel is sharded on both the `fsdp` and `fsdp_transpose` mesh axes,
    a single combined 2-axis all-gather forces XLA to materialize an interleave
    transpose to fix the layout. Splitting into two single-axis gathers separated
    by an `optimization_barrier` makes each stage produce a contiguous layout, so
    no transpose is emitted. Mirrors `moe_fsdp_use_two_stage_all_gather`.
    """
    if (
        not self.use_two_stage_all_gather
        or self.mesh is None
        or self.mesh.shape.get("fsdp", 1) <= 1
        or self.mesh.shape.get("fsdp_transpose", 1) <= 1
    ):
      return kernel

    # kernel_axes is a plain tuple of logical names; wrap it so the logical-to-physical
    # lookup treats it as a single spec rather than a pytree of strings.
    full_logical = PartitionSpec(*self.kernel_axes)
    # Stage 1 gathers fsdp_transpose, stage 2 gathers the remaining fsdp.
    stage1 = get_physical_spec_without_axes(full_logical, self.mesh, ("fsdp_transpose",))
    stage2 = get_physical_spec_without_axes(full_logical, self.mesh, FSDP_MESH_AXES)
    if stage1.spec == stage2.spec:
      # Not sharded on both FSDP axes, so a single all-gather is already optimal.
      return kernel

    shard = functools.partial(maybe_shard_with_name, shard_mode=self.shard_mode, debug_sharding=self.debug_sharding)
    kernel = shard(kernel, stage1)
    kernel = jax.lax.optimization_barrier(kernel)
    kernel = shard(kernel, stage2)
    return kernel

  def _maybe_hoist_weight_ag(self, kernel):
    """Issue the FSDP weight all-gather explicitly, tagged, ahead of the dot.

    By default SPMD inserts this gather itself, right next to the matmul that
    consumes the kernel, and it lands on the *quantized* tensor inside
    `_compute_dot_general_nnx`. For the MoE shared expert that gather measures
    7.5 ms/iter at ~1.1 GB/s in the forward, while a comparable backward gather
    of more bytes runs at 29.9 GB/s, so the cost tracks the phase rather than
    the payload and the scheduler is what we want to reach.

    A sharding constraint does NOT work here, and the AOT dump says so: dropping
    the fsdp axes with `maybe_shard_with_name` did move the gather (the
    shared-expert `f8e4m3fn[1,7168,2048]` became `bf16[1,7168,2048]`, total
    all-gather count 236 -> 235, so a gather moved rather than one being added),
    but zero ops carried the id. SPMD absorbs the constraint and emits a fresh
    all-gather of its own, still marked `is_spmd_generated="true"` and still under
    `closed_call/convert_element_type`, and our frontend attribute does not
    survive that substitution. That bought 2x the wire bytes and no scheduling
    control.

    So emit the collective ourselves, the way `moe.py`'s `_make_cv_gather` does
    for the routed experts: a `shard_map` around `lax.all_gather` is a real op in
    the jaxpr that SPMD does not replace, so the tag stays attached to it. The
    `custom_vjp` keeps the PRIMAL gather unannotated, which is what
    `remat_policy=custom` recomputes in the backward; annotating the backward
    gather back-edges its reduce-scatter into the rematerialized forward and
    raises a FAILED_PRECONDITION scheduling cycle. The backward rule is the
    transpose of a tiled all-gather, a tiled psum_scatter, so FSDP weight grads
    stay correct and no large residual is saved.

    The kernel is still bf16 at this point, so the wire cost doubles relative to
    the e4m3 gather SPMD would emit (29.4 MB vs 14.7 MB for the shared expert).
    The backward measurement above says that is affordable if placement is what
    is broken. Quantizing here instead to keep e4m3 would mean handing a
    pre-quantized QArray to `quant_dot_general`, which risks double-quantization;
    the routed-expert notes record that the tagged variant of exactly that
    machinery NaN'd on cluster round 2 and lost 5.302 vs 5.106.
    """
    sg = self.hoist_weight_ag_sched_group
    if sg is None or sg < 0 or self.mesh is None:
      return kernel
    if not any(self.mesh.shape.get(ax, 1) > 1 for ax in FSDP_MESH_AXES):
      return kernel

    full_logical = PartitionSpec(*self.kernel_axes)
    in_spec = get_physical_spec_without_axes(full_logical, self.mesh, ()).spec
    out_spec = get_physical_spec_without_axes(full_logical, self.mesh, FSDP_MESH_AXES).spec
    if in_spec == out_spec:
      return kernel  # nothing sharded on an FSDP axis, so there is no gather to hoist

    diff = [i for i in range(len(in_spec)) if in_spec[i] != out_spec[i]]
    if len(diff) != 1:
      # More than one dim changes only under 2D FSDP, which `_maybe_two_stage_all_gather`
      # already handles; leave those to SPMD rather than guess a single gather axis.
      return kernel
    gather_axis = diff[0]
    names = in_spec[gather_axis]
    names = (names,) if isinstance(names, str) else tuple(names)
    ag_axes = tuple(n for n in names if n in FSDP_MESH_AXES)
    if not ag_axes:
      return kernel

    @jax.custom_vjp
    def _gather(w):  # PRIMAL: plain gather, i.e. what remat recomputes in the backward
      return jax.shard_map(
          lambda x: jax.lax.all_gather(x, ag_axes, axis=gather_axis, tiled=True),
          mesh=self.mesh,
          in_specs=(in_spec,),
          out_specs=out_spec,
          check_vma=False,
      )(w)

    def _gather_fwd(w):  # FORWARD under diff: annotated, so it can be co-scheduled
      def _fn(x):
        with xla_metadata.set_xla_metadata(_scheduling_group_id=sg):
          return jax.lax.all_gather(x, ag_axes, axis=gather_axis, tiled=True)

      w_full = jax.shard_map(
          _fn, mesh=self.mesh, in_specs=(in_spec,), out_specs=out_spec, check_vma=False
      )(w)
      return w_full, None  # no residual: the sharded param is a leaf, always available

    def _gather_bwd(_res, ct):  # transpose of a tiled all-gather = tiled psum_scatter
      g = jax.shard_map(
          lambda gg: jax.lax.psum_scatter(gg, ag_axes, scatter_dimension=gather_axis, tiled=True),
          mesh=self.mesh,
          in_specs=(out_spec,),
          out_specs=in_spec,
          check_vma=False,
      )(ct)
      return (g,)

    _gather.defvjp(_gather_fwd, _gather_bwd)
    return _gather(kernel)

  def __call__(
      self,
      inputs: Array,
      _initializing: bool = False,
      out_sharding: NamedSharding | None = None,
      slice_bounds: tuple[int, int] | None = None,
  ) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.
      _initializing: Whether the module is initializing.
      out_sharding: Optional sharding for the output.
      slice_bounds: Optional tuple (begin, end) to slice the kernel and bias on
        the last (output-feature) axis before contraction. Unquantized only.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = normalize_axes(self.axis, inputs.ndim)

    for i, ax in enumerate(norm_axis):
      if inputs.shape[ax] != self.in_features_shape[i]:
        raise ValueError(
            f"Input dimension {inputs.shape[ax]} at axis {ax} "
            f"does not match expected input feature size {self.in_features_shape[i]}"
        )

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = getattr(self.kernel, "value", self.kernel)
      if hasattr(kernel, "value"):
        kernel = kernel.value
      # Move logit_dense kernel to device if parameter offloading is enabled
      if self.parameter_memory_host_offload:
        max_logging.log("linear.py: Moving parameter logits_dense kernel to device")
        kernel = jax.device_put(kernel, max_utils.device_space())
      # NOTE: do NOT tag or checkpoint_name here. This is the SHARDED parameter, upstream of both
      # the qwix quantize and the SPMD weight all-gather, which happen together inside
      # `_compute_dot_general_nnx` below. A `_scheduling_group_id` placed on this cast lands on a
      # no-op (the param is already bf16), XLA folds the cast away, and the annotation goes with it
      # -- measured: 2 ops carried the id in the optimized HLO, neither of them a gather. A
      # `checkpoint_name` here names a leaf that is always available, so it cannot stop the backward
      # from re-running the gather. The hoist below is the site that actually reaches the gather.
      kernel = jnp.asarray(kernel, self.dtype)

    if slice_bounds is not None:
      if self.quant is not None:
        raise ValueError("sliced contraction is only supported when quant is None")
      begin, end = slice_bounds
      if not 0 <= begin < end <= kernel.shape[-1]:
        raise ValueError(f"slice_bounds {slice_bounds} must be valid and within [0, {kernel.shape[-1]}]")
      kernel = kernel[..., begin:end]

    kernel = self._maybe_two_stage_all_gather(kernel)
    kernel = self._maybe_hoist_weight_ag(kernel)
    # STEP 0 PROBE: does a start/done PAIR survive remat? Every weight in the census appears
    # as a fwd + bwd-remat pair, so if the pair re-traces into rematted_computation, one
    # forward-side change covers both members. Placement probe only -- split_copy moves the
    # same bytes an identity would.
    if _STARTDONE_PROBE and self.hoist_weight_ag_sched_group is not None \
        and self.hoist_weight_ag_sched_group >= 0:
      from maxtext.kernels.startdone import split_copy

      _pspec = get_physical_spec_without_axes(
          PartitionSpec(*self.kernel_axes), self.mesh, FSDP_MESH_AXES).spec
      kernel = split_copy(kernel, self.mesh, _pspec)

    # out_sharding should be None for auto mesh axis
    if self.shard_mode != ShardMode.EXPLICIT:
      out_sharding = None

    contract_ind = tuple(range(0, len(self.axis)))
    output = _compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general if slice_bounds is None else None,
        _initializing,
        out_sharding,
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias[...], self.dtype)
      if slice_bounds is not None:
        begin, end = slice_bounds
        bias = bias[..., begin:end]
      output += bias
    return output


def dense_general(
    *,
    inputs_shape: tuple[int, ...] | None = None,
    in_features_shape: tuple[int, ...] | int | None = None,
    out_features_shape: Iterable[int] | int,
    axis: Iterable[int] | int = -1,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[None | str, ...] = (),
    quant: None | Quant = None,
    use_bias: bool = False,
    shard_mode: ShardMode = ShardMode.AUTO,
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    name: None | str = None,
):
  """Creates a DenseGeneral Linen module using nnx.bridge.to_linen.

  Args:
    inputs_shape: tuple with the shape of the inputs
    in_features_shape: tuple with numbers of input features for axes specified in
      'axis'.
    out_features_shape: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    weight_dtype: the dtype of the weights (default: float32).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    kernel_axes: logical axes for partitioning the kernel.
    quant: quantization config, defaults to None implying no quantization.
    use_bias: whether to add bias in linear transformation.
    shard_mode: indicating the shard mode
    matmul_precision: Precision for matrix multiplication.
    parameter_memory_host_offload: Determines whether to offload params to host
    name: name passed to the ToLinen Module
  """
  if not (inputs_shape is not None) ^ (in_features_shape is not None):
    raise ValueError("Exactly one of inputs_shape or in_features must be specified.")

  if inputs_shape is not None:
    axis = canonicalize_tuple(axis)
    in_features_shape = tuple(inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape)))
  else:
    assert in_features_shape is not None
  module = nnx_wrappers.to_linen(
      DenseGeneral,
      in_features_shape=in_features_shape,
      out_features_shape=out_features_shape,
      axis=axis,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      quant=quant,
      use_bias=use_bias,
      shard_mode=shard_mode,
      matmul_precision=matmul_precision,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


class Dropout(nnx.Dropout):
  """Forked nnx.Dropout that is easier to use with bridge"""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      rate: float,
      *,
      broadcast_dims: Sequence[int] = (),
      deterministic: bool = False,
      rng_collection: str = "dropout",
      rngs: nnx.Rngs | None = None,
  ):
    self.rate = rate
    self.broadcast_dims = broadcast_dims
    self.deterministic = deterministic
    self.rng_collection = rng_collection

    if isinstance(rngs, nnx.Rngs):
      self.rngs = rngs.fork() if hasattr(type(rngs), "fork") else rngs
    else:
      raise TypeError(f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}.")


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      in_features: int,
      intermediate_dim: int = 2048,
      activations: Sequence[str | Callable[..., Any]] = ("relu",),
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      intermediate_dropout_rate: float = 0.1,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      use_bias: bool = False,
      use_pre_norm: bool = False,
      quant: None | Quant = None,
      model_mode: None | str = None,
      is_shared_expert: bool = False,
      *,
      rngs: nnx.Rngs,
  ) -> None:
    """A MlpBlock module.

    Args:
      config: Config object containing model parameters.
      mesh: Mesh object of device and physical axes information
      in_features: Number of input features.
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: computation data type for the dense layer.
      weight_dtype: weight data type for the dense layer.
      use_bias: whether to add bias in all feedforward layers.
      use_pre_norm: whether to add pre layer norm in mlp layers.
      quant: Optional quantization config, no quantization if None.
      out_sharding: Named sharding of outputs
    """
    self._is_shared_expert = is_shared_expert
    self.config = config
    self.mesh = mesh
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.activations = activations
    self.kernel_init = kernel_init
    self.intermediate_dropout_rate = intermediate_dropout_rate
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.use_bias = use_bias
    self.use_pre_norm = use_pre_norm
    self.quant = quant
    self.model_mode = model_mode

    if self.use_pre_norm:
      self.mlp_layer_norm = self.get_norm_layer(num_features=in_features)(
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=config.normalization_layer_epsilon,
          rngs=rngs,
      )
    else:
      self.mlp_layer_norm = None

    if self.model_mode == MODEL_MODE_PREFILL:
      self.intermediate_logical = ("activation_batch", "prefill_activation_length", "activation_mlp")
    else:
      self.intermediate_logical = ("activation_batch", "activation_length", "activation_mlp")

    if config.fused_mlp:
      self.wi = DenseGeneral(
          in_features_shape=in_features,
          out_features_shape=(len(self.activations), self.intermediate_dim),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "num_activations", "mlp"),
          hoist_weight_ag_sched_group=self._hoist_wag_sg(0),
          quant=self.quant,
          use_bias=self.use_bias,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          mesh=self.mesh,
          use_two_stage_all_gather=self.config.dense_fsdp_use_two_stage_all_gather,
          debug_sharding=self.config.debug_sharding,
          rngs=rngs,
      )
    else:
      for idx in range(len(self.activations)):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = DenseGeneral(
            in_features_shape=in_features,
            out_features_shape=self.intermediate_dim,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self._wi_kernel_axes(),
            hoist_weight_ag_sched_group=self._hoist_wag_sg(idx),
            quant=self.quant,
            use_bias=self.use_bias,
            shard_mode=self.config.shard_mode,
            matmul_precision=self.config.matmul_precision,
            mesh=self.mesh,
            use_two_stage_all_gather=self.config.dense_fsdp_use_two_stage_all_gather,
            debug_sharding=self.config.debug_sharding,
            rngs=rngs,
        )
        setattr(self, dense_name, module)
    self.dropout = Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,), rngs=rngs)
    self.wo = DenseGeneral(
        in_features_shape=self.intermediate_dim,
        out_features_shape=in_features,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=self.kernel_init,
        kernel_axes=self._wo_kernel_axes(),
        hoist_weight_ag_sched_group=self._hoist_wag_sg(2),
        quant=self.quant,
        use_bias=self.use_bias,
        shard_mode=self.config.shard_mode,
        matmul_precision=self.config.matmul_precision,
        mesh=self.mesh,
        use_two_stage_all_gather=self.config.dense_fsdp_use_two_stage_all_gather,
        debug_sharding=self.config.debug_sharding,
        rngs=rngs,
    )

    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

  def get_norm_layer(self, num_features: int):
    """get normalization layer."""
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.QWEN3,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(normalizations.RMSNorm, num_features=num_features)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      from maxtext.models import gpt3  # pylint: disable=import-outside-toplevel

      return functools.partial(
          gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=self.use_bias
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def _replicate_embed(self) -> bool:
    """Whether this block's kernels should leave the embed axis unsharded.

    Only the MoE shared expert opts in. Its [d_model, d_ff_shared] kernel is small (14.7 MB), but
    sharding embed over fsdp=128 leaves 56 rows per device, and the resulting SPMD all-gather is a
    127-hop ring carrying ~114 KB per hop -- measured 7.5 ms/iter at ~1% link utilization.
    Replicating the weight deletes that collective at a few hundred MB of per-device storage.
    """
    return bool(getattr(self.config, "moe_shared_expert_replicate", False)) and bool(
        getattr(self, "_is_shared_expert", False)
    )

  def _hoist_wag_sg(self, offset=0):
    """Scheduling-group id for this block's explicit FSDP weight all-gather.

    Only the MoE shared expert opts in, so the MLA projections and the logits dense keep the
    stock SPMD-inserted gather. Each weight gets its own id (base, +1, +2) so the all-gather
    combiner cannot fuse the three into one monolith that no single compute region can hide,
    which is the same reasoning the routed-expert gathers use in `moe.py`.
    """
    base = getattr(self.config, "shared_expert_weight_ag_sched_group", -1)
    if base is None or base < 0 or not getattr(self, "_is_shared_expert", False):
      return -1
    return base + offset

  def _wi_kernel_axes(self):
    return (None, "mlp") if self._replicate_embed() else ("embed", "mlp")

  def _wo_kernel_axes(self):
    return ("mlp", None) if self._replicate_embed() else ("mlp", "embed")

  def __call__(
      self,
      inputs,
      decode: bool = False,
      deterministic: bool = False,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
  ):
    """Applies Transformer MlpBlock module."""
    cfg = self.config

    if self.mlp_layer_norm is not None:
      inputs = self.mlp_layer_norm(inputs)

    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    if cfg.fused_mlp:
      x = self.wi(inputs, out_sharding=intermediate_sharding)

      # Enforce fused activations don't shard on num_activations axis
      fused_intermediate_logical = self.intermediate_logical[:2] + (None,) + self.intermediate_logical[2:]
      x = self._maybe_shard_with_logical(x, fused_intermediate_logical)

      x = checkpoint_name(x, "mlpwi")
      for idx, act_fn in enumerate(self.activations):
        y = _convert_to_activation_function(act_fn)(x[:, :, idx, ...])
        activations.append(y)
    else:
      for idx, act_fn in enumerate(self.activations):
        dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
        module = getattr(self, dense_name)
        x = module(inputs, out_sharding=intermediate_sharding)
        x = checkpoint_name(x, "mlp" + dense_name)
        if cfg.activations_in_float32:
          x = x.astype(jnp.float32)
        x = _convert_to_activation_function(act_fn)(x)
        activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations).astype(self.dtype)
    # Apply dropout and final dense output projection.
    x = self.dropout(x, deterministic=deterministic)  # Broadcast along length.
    x = self._maybe_shard_with_logical(x, self.intermediate_logical)
    output = self.wo(x, out_sharding=out_sharding)

    output = checkpoint_name(output, "mlpwo")
    return output


def mlp_block(
    *,
    config: Config,
    mesh: Mesh,
    in_features: int,
    intermediate_dim: int = 2048,
    activations: Sequence[str | Callable[..., Any]] = ("relu",),
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    intermediate_dropout_rate: float = 0.1,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    use_bias: bool = False,
    use_pre_norm: bool = False,
    quant: None | Quant = None,
    model_mode: None | str = None,
    name: None | str = None,
):
  """Creates a MlpBlock Linen module using nnx.bridge.to_linen."""
  module = nnx_wrappers.to_linen(
      MlpBlock,
      config=config,
      mesh=mesh,
      in_features=in_features,
      intermediate_dim=intermediate_dim,
      activations=activations,
      kernel_init=kernel_init,
      intermediate_dropout_rate=intermediate_dropout_rate,
      dtype=dtype,
      weight_dtype=weight_dtype,
      use_bias=use_bias,
      use_pre_norm=use_pre_norm,
      quant=quant,
      model_mode=model_mode,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module


class DeepSeekV4GroupedLinear(nnx.Module):
  """Block-diagonal grouped linear used by the grouped output projection in DeepSeek-V4.

  The core attention's stacked output is `num_attention_heads * head_dim`-dim,
  which is extremely large. A direct projection would dominate the per-token cost.
  This module splits the heads into `g` groups, projecting each independently
  to a smaller intermediate dimension, which are later mixed.
  """

  def __init__(
      self,
      in_features_per_group: int,
      out_features: int,
      n_groups: int,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: tuple[None | str, ...] = ("groups", "embed", "mlp"),
      matmul_precision: str = "default",
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the DeepSeekV4GroupedLinear module.

    Args:
      in_features_per_group: The size of the input dimension for each group.
      out_features: The total output dimension across all groups. Must be divisible by n_groups.
      n_groups: The number of independent groups to split the projection into.
      weight_dtype: the dtype of the weights (default: float32).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
      kernel_axes: logical axes for partitioning the kernel.
      matmul_precision: Precision for matrix multiplication.
      parameter_memory_host_offload: Determines whether to offload params to host
      rngs: RNG state for initialization in nnx.
    """
    if out_features % n_groups != 0:
      raise ValueError(f"out_features ({out_features}) must be divisible by n_groups ({n_groups})")

    self.in_features_per_group = in_features_per_group
    self.out_features = out_features
    self.n_groups = n_groups
    self.out_features_per_group = out_features // n_groups

    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.matmul_precision = matmul_precision
    self.parameter_memory_host_offload = parameter_memory_host_offload

    # Kernel shape splits the projection up into a batched representation
    kernel_shape = (self.n_groups, self.in_features_per_group, self.out_features_per_group)

    # NdInitializer takes tuple positions to calculate fan_in / fan_out.
    # Axis 1 represents the inner contracting dimension (fan_in).
    # Axis 2 represents the output features dimension (fan_out).
    kernel_in_axis = (1,)
    kernel_out_axis = (2,)

    self.kernel = nnx.Param(
        self.kernel_init(
            rngs.params(),
            kernel_shape,
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
        ),
        sharding=self.kernel_axes,
    )

  def __call__(self, inputs: Array) -> Array:
    """Applies a batched grouped linear transformation to the inputs.

    Args:
      inputs: The nd-array to be transformed. Expected shape is `[..., n_groups, in_features_per_group]`.

    Returns:
      The transformed input of shape `[..., n_groups, out_features_per_group]`.
      When later flattened across the last two dims, this results in `out_features`.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    kernel = self.kernel[...]
    if self.parameter_memory_host_offload:
      max_logging.log("linear.py: Moving parameter grouped_linear kernel to device")
      kernel = jax.device_put(kernel, max_utils.device_space())
    kernel = jnp.asarray(kernel, self.dtype)

    # Perform a batched matrix multiplication using einsum with explicit precision.
    # We use jnp.einsum instead of explicitly flattening and using lax.dot_general
    # to make the group-wise broadcast highly readable and natively batched.
    #
    # Notation breakdown:
    #   ... : Any leading batch/sequence dimensions (e.g., [Batch, SeqLen]).
    #   g   : The n_groups dimension.
    #   i   : The in_features_per_group dimension (the contracting dimension).
    #   o   : The out_features_per_group dimension.
    #
    # Input shape:  [..., g, i]
    # Kernel shape: [g, i, o]
    # Output shape: [..., g, o]
    output = jnp.einsum("...gi,gio->...go", inputs, kernel, precision=lax.Precision(self.matmul_precision))

    return output


def deepseek_v4_grouped_linear(
    *,
    in_features_per_group: int,
    out_features: int,
    n_groups: int,
    weight_dtype: DType = jnp.float32,
    dtype: DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: tuple[None | str, ...] = ("groups", "embed", "mlp"),
    matmul_precision: str = "default",
    parameter_memory_host_offload: bool = False,
    name: None | str = None,
):
  """Creates a DeepSeekV4GroupedLinear Linen module using nnx.bridge.to_linen."""
  module = nnx_wrappers.to_linen(
      DeepSeekV4GroupedLinear,
      in_features_per_group=in_features_per_group,
      out_features=out_features,
      n_groups=n_groups,
      weight_dtype=weight_dtype,
      dtype=dtype,
      kernel_init=kernel_init,
      kernel_axes=kernel_axes,
      matmul_precision=matmul_precision,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
  return module
