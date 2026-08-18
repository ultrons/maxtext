"""Split-phase (start/done) DMA wrappers.

`start` arms a DMA and returns without waiting, so XLA can schedule real compute after it.
`done` RECONSTRUCTS the identical descriptor and waits. Both are ordinary TC-shaped custom
calls, so XLA schedules around them rather than fencing at them.

No semaphore crosses the kernel boundary. Each half allocates its own DMA semaphore as
kernel scratch, which is what places it in sync-flag memory; because Mosaic's allocation is
deterministic, `done`'s rebuilt descriptor names the sync flag `start` armed.

HARD RULE, measured on v7x (probes/startdone_runtime.py, gates R5/R6): the two halves must
declare BYTE-IDENTICAL `scratch_shapes`. Under a differing scratch footprint Mosaic assigns
the DMA semaphore a different slot, the wait targets the wrong flag, and it HANGS -- it does
not raise. `_SCRATCH` below is shared by both halves for exactly this reason, and
`assert_scratch_matches()` is the guard so a future edit cannot break it silently.
"""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HBM = pl.BlockSpec(memory_space=pltpu.HBM)

# The ONE scratch signature. Both halves must use this exact list (see R5/R6).
_SCRATCH = [pltpu.SemaphoreType.DMA]


def assert_scratch_matches(a, b):
  """Guard the R5 hang: identical scratch signatures on both halves."""
  if list(a) != list(b):
    raise ValueError(
        "start/done scratch_shapes must be byte-identical; mismatched footprints make "
        f"Mosaic slot the DMA semaphore differently and the wait HANGS. got {a} vs {b}"
    )


def _start_body(x_ref, o_ref, sem):
  pltpu.make_async_copy(x_ref, o_ref, sem).start()


def _done_body(x_ref, d_ref, o_ref, sem):
  # d_ref is aliased to the output, so nothing is loaded here (Mosaic forbids HBM loads).
  pltpu.make_async_copy(x_ref, d_ref, sem).wait()


def _split_copy_impl(x, mesh, spec):
  """Identity, expressed as an armed DMA and a separate wait.

  Structurally this is what a split-phase gather looks like from XLA's point of view: two
  TC custom calls with a real buffer dependency between them, and room for compute in the
  gap. It moves the same bytes a plain copy would, so it is a placement probe rather than a
  performance change.
  """
  assert_scratch_matches(_SCRATCH, _SCRATCH)
  shard = tuple(
      d // mesh.shape[a] if (a := spec[i]) is not None and a in mesh.shape else d
      for i, d in enumerate(x.shape)
  )
  shape = jax.ShapeDtypeStruct(shard, x.dtype)
  start = pl.pallas_call(
      _start_body, in_specs=[_HBM], out_specs=_HBM, out_shape=shape,
      scratch_shapes=_SCRATCH,
  )
  done = pl.pallas_call(
      _done_body, in_specs=[_HBM, _HBM], out_specs=_HBM, out_shape=shape,
      scratch_shapes=_SCRATCH, input_output_aliases={1: 0},
  )
  # Mosaic kernels cannot be automatically partitioned ("Please wrap the call in a
  # shard_map"), so the pair runs per-shard with the kernel's own physical spec.
  return jax.shard_map(
      lambda xx: done(xx, start(xx)), mesh=mesh,
      in_specs=(spec,), out_specs=spec, check_vma=False,
  )(x)


# A pallas_call on a grad-live path cannot be differentiated: autodiff descends into the
# kernel and Pallas's JVP rule asserts (`_pallas_call_jvp_rule` -> `ad.jvp_jaxpr` ->
# AssertionError). Wrap it so autodiff never enters, and supply the transpose ourselves --
# the same reason `moe.py`'s `_make_cv_gather` is a custom_vjp. The PRIMAL keeps the real
# start/done pair, so `remat_policy=custom` re-runs the pair in the backward, which is the
# property this probe exists to measure.
@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def split_copy(x, mesh, spec):
  return _split_copy_impl(x, mesh, spec)


def _split_copy_fwd(x, mesh, spec):
  return _split_copy_impl(x, mesh, spec), None


def _split_copy_bwd(mesh, spec, _res, ct):
  # Transpose of an identity copy is the identity. A real gather's transpose is a tiled
  # psum_scatter (see `_make_cv_gather`).
  return (ct,)


split_copy.defvjp(_split_copy_fwd, _split_copy_bwd)


# =============================================================================
# Forward split-phase all-gather.
#
# Push model: every device arms one DMA per peer, writing its OWN shard into that peer's
# output slot, plus a local copy into its own slot. `start` arms all S of them and returns;
# `done` reconstructs the identical descriptors and waits. One start, one done, maximal
# concurrency, and the layer's compute sits in the gap.
#
# The output is shaped (S,) + shard so the destination slot is a clean `o_ref.at[me]`;
# callers reshape to the gathered layout afterwards.
#
# The BACKWARD is `lax.psum_scatter`, i.e. XLA's reduce-scatter, deliberately. Measured on
# the 6.864 s profile the reduce-scatters run 10.4-25.9 GB/s while the gathers we are
# chasing run 0.3-1.1 GB/s, and `rs-lever-mapped-closed` already found direct-to-owner RS
# near-optimal and not reclaimable by a better kernel. The RS is not the pathology, so we do
# not write an accumulating kernel to replace it.
#
# Barrier: a peer must not write into our output before it exists, so both halves open with
# a `get_barrier_semaphore` handshake keyed by `collective_id`.
# =============================================================================

# start and done share this signature -- see R5/R6, a mismatch HANGS rather than raising.
_AG_SCRATCH = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]


def _peer_id(axis_name, mesh_axes, d):
  """Rank `d` on the gather axis, every other mesh axis held at this device's index."""
  return {a: (d if a == axis_name else jax.lax.axis_index(a)) for a in mesh_axes}


def _ag_descriptors(x_ref, o_ref, loc, ss, rs, axis_name, mesh_axes, n):
  """The S copies, built identically by both halves. Order matters: it fixes the slots.

  ROTATION, not `for d in range(n)`. The loop is static but `me` is dynamic, so a plain
  range includes d == me and the device both local-copies AND remote-copies into its own
  slot: two writes to one destination and a semaphore count that does not match the waits.
  That halts the core (`RuntimeUnexpectedCoreHalt`). Stepping `peer = (me + step) % n` over
  step in 1..n-1 visits every OTHER device exactly once, with no self-send.
  """
  me = jax.lax.axis_index(axis_name)
  yield pltpu.make_async_copy(x_ref, o_ref.at[me], loc)
  for step in range(1, n):
    peer = jax.lax.rem(me + step, n)
    yield pltpu.make_async_remote_copy(
        x_ref, o_ref.at[me], ss, rs, device_id=_peer_id(axis_name, mesh_axes, peer)
    )


def _ag_barrier(axis_name, mesh_axes, n):
  """Handshake so no peer writes into our output before it exists.

  device_id must be the multi-axis MESH dict, not a bare int: a flat integer is only
  correct on a 1-D mesh and silently addresses the wrong device on the model's mesh.
  """
  bar = pltpu.get_barrier_semaphore()
  me = jax.lax.axis_index(axis_name)
  for step in range(n):
    peer = jax.lax.rem(me + step, n)
    pl.semaphore_signal(bar, device_id=_peer_id(axis_name, mesh_axes, peer))
  pl.semaphore_wait(bar, n)


def make_ag_bodies(axis_name, mesh_axes, n):
  """Build the start/done kernel bodies for an S-way push all-gather."""

  def start_body(x_ref, o_ref, loc, ss, rs):
    _ag_barrier(axis_name, mesh_axes, n)
    for dma in _ag_descriptors(x_ref, o_ref, loc, ss, rs, axis_name, mesh_axes, n):
      dma.start()

  def done_body(x_ref, d_ref, o_ref, loc, ss, rs):
    for dma in _ag_descriptors(x_ref, d_ref, loc, ss, rs, axis_name, mesh_axes, n):
      dma.wait()

  return start_body, done_body


_AG_CP = pltpu.CompilerParams(
    collective_id=7, allow_collective_id_without_custom_barrier=True, has_side_effects=True
)


def _sag_impl(w, mesh, axis_name, in_spec, out_spec):
  n = mesh.shape[axis_name]
  mesh_axes = tuple(mesh.axis_names)
  start_body, done_body = make_ag_bodies(axis_name, mesh_axes, n)
  assert_scratch_matches(_AG_SCRATCH, _AG_SCRATCH)

  def body(xx):
    shard = jax.ShapeDtypeStruct((n,) + xx.shape, xx.dtype)
    start = pl.pallas_call(
        start_body, in_specs=[_HBM], out_specs=_HBM, out_shape=shard,
        scratch_shapes=_AG_SCRATCH, compiler_params=_AG_CP,
    )
    done = pl.pallas_call(
        done_body, in_specs=[_HBM, _HBM], out_specs=_HBM, out_shape=shard,
        scratch_shapes=_AG_SCRATCH, input_output_aliases={1: 0}, compiler_params=_AG_CP,
    )
    g = done(xx, start(xx))               # <-- layer compute belongs in this gap
    return g.reshape((n * xx.shape[0],) + xx.shape[1:])

  return jax.shard_map(
      body, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec, check_vma=False
  )(w)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def split_all_gather(w, mesh, axis_name, in_spec, out_spec):
  """FSDP weight all-gather we own the placement of. Backward is XLA's reduce-scatter."""
  return _sag_impl(w, mesh, axis_name, in_spec, out_spec)


def _sag_fwd(w, mesh, axis_name, in_spec, out_spec):
  return _sag_impl(w, mesh, axis_name, in_spec, out_spec), None


def _sag_bwd(mesh, axis_name, in_spec, out_spec, _res, ct):
  # Transpose of a tiled all-gather is a tiled psum_scatter. Left to XLA on purpose: the
  # reduce-scatters measure 10.4-25.9 GB/s against the gathers' 0.3-1.1 GB/s, so the RS is
  # not the pathology and an accumulating kernel would be effort on the healthy collective.
  g = jax.shard_map(
      lambda c: jax.lax.psum_scatter(c, axis_name, scatter_dimension=0, tiled=True),
      mesh=mesh, in_specs=(out_spec,), out_specs=in_spec, check_vma=False,
  )(ct)
  return (g,)


split_all_gather.defvjp(_sag_fwd, _sag_bwd)
