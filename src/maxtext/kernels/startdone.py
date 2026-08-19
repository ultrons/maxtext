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
import jax.numpy as jnp
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
# Forward split-phase all-gather -- STATIC destinations only.
#
# The natural push formulation (remote copy into `o_ref.at[me]`) is REFUTED on hardware:
# the destination offset is computed on the sender and must be interpreted in the
# receiver's buffer, and the bytes silently never arrive (probes/ag_ladder.py, all slots
# empty). A pull model has the same defect. What DOES deliver (probes/ag_static.py, PASS)
# is a remote copy into a STATIC whole-buffer destination, so this design never indexes:
#
#   at step k (k = 1..n-1, a PYTHON constant) device `me` pushes its shard into device
#   (me+k)'s buffer k.
#
# k is static on both sides. Receiver j's buffer k holds shard (j-k) mod n; the gathered
# array is assembled OUTSIDE the kernel with ordinary XLA ops (a dynamic take + reshape,
# ~8 us of HBM traffic for a 29 MB weight -- no DMA addressing involved).
#
# `start` arms all n-1 sends and returns; `done` reconstructs the identical descriptors
# and waits. Both halves share ONE scratch signature (R5/R6: a mismatch HANGS). Each step
# has its own send/recv semaphore pair, sidestepping the untested question of whether
# n-1 in-flight copies may share one pair.
#
# The BACKWARD is `lax.psum_scatter` -- XLA's reduce-scatter -- deliberately. Measured on
# the 6.864 s profile the reduce-scatters run 10.4-25.9 GB/s while the gathers we are
# chasing run 0.3-1.1 GB/s; `rs-lever-mapped-closed` found direct-to-owner RS
# near-optimal. The RS is not the pathology. The pathological backward item `.445` is the
# forward gather RE-RUN inside rematted_computation, and Step 0 proved this pair
# re-traces there intact, so the forward conversion covers it.
# =============================================================================

def _ag_cp(slot):
  # Distinct collective_id per pair: the entry barrier is keyed by it, and co-scheduled
  # pairs sharing one barrier counter alias exactly like the DMA semaphores did.
  return pltpu.CompilerParams(
      collective_id=7 + slot, allow_collective_id_without_custom_barrier=True,
      has_side_effects=True,
  )


def _peer_id(axis_name, mesh_axes, d):
  """Multi-axis mesh device id: rank `d` on the gather axis, every other axis held fixed.

  A bare int is only correct on a 1-D mesh; the model mesh is multi-axis.
  """
  return {a: (d if a == axis_name else jax.lax.axis_index(a)) for a in mesh_axes}


def _make_ag_pair(n, shard_sds, axis_name, mesh_axes, slot=0):
  """Build the start/done pallas_call pair for an n-way static-destination all-gather.

  `slot` separates CO-SCHEDULED pairs. Scratch semaphores are allocated at the same
  physical sync-flag slots for every kernel with the same signature, and DMA semaphores
  are anonymous counters -- so when XLA interleaves two pairs' starts before their dones
  (the model runs three pairs per layer), pair B's completions satisfy pair A's waits.
  REPRODUCED at n=8 (probe R9, same halt signature as sag1d). `slot` leading dummy
  REGULAR semaphores displace this pair's DMA semaphores to distinct physical flags.
  """
  # ONE shared (send, recv) DMA semaphore pair for ALL n-1 sends -- constant in n.
  # Per-step PAIRS halted at n=128: 2*(n-1) = 254 scratch semaphores overflows the
  # sync-flag budget ("Semaphore (scratch argument 253) has a nonzero value upon exit",
  # siv-cn-sag1b, zero steps), invisible at the rig's 14 and at compile time. DMA
  # semaphores are counters, so concurrent copies sharing a pair is legal: each
  # descriptor's wait decrements its own byte count (the ladder's shared-pair rungs
  # passed). Buffers stay per-step -- STATIC destinations are what correctness requires.
  # IDENTICAL in both halves (R5/R6).
  # [send, recv, exit]. The REGULAR `exit` semaphore is the EPOCH FENCE: the pair runs
  # 116x/step under scan+remat, every instance reuses the same physical sync-flag slots,
  # and DMA semaphores are anonymous counters -- so without a fence a fast device can pass
  # the entry barrier for execution i+1 on a laggard's execution-i signals and arm sends
  # into a peer still waiting in done_i, which consumes the wrong epoch's bytes and leaves
  # the residue the halt reported ("Semaphore (scratch argument 0) has a nonzero value",
  # sag1c). done therefore ends with signal-all + wait-n on `exit`: a device cannot
  # complete its i-th exit wait until every device has signaled its i-th, because each
  # device's cumulative signals are bounded by its own completed dones. Hard serialization
  # of executions; skew across epochs becomes impossible rather than unlikely.
  # Pad WITHIN EACH POOL: Mosaic allocates DMA and REGULAR semaphores separately, so
  # REGULAR-only padding left the DMA slots shared (R9 still halted with it). Layout:
  # [2*slot dummy DMA] [ss] [rs] [slot dummy REGULAR] [exit].
  scratch = (
      [pltpu.SemaphoreType.DMA] * (2 * slot)
      + [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
      + [pltpu.SemaphoreType.REGULAR] * slot
      + [pltpu.SemaphoreType.REGULAR]
  )

  def descriptors(x_ref, bufs, sems):
    ss, rs = sems[2 * slot], sems[2 * slot + 1]
    me = jax.lax.axis_index(axis_name)
    for i, k in enumerate(range(1, n)):
      peer = jax.lax.rem(me + k, n)
      yield pltpu.make_async_remote_copy(
          x_ref, bufs[i], ss, rs,
          device_id=_peer_id(axis_name, mesh_axes, peer),
      )

  def start_body(x_ref, *rest):
    bufs, sems = list(rest[: n - 1]), list(rest[n - 1:])
    # A peer must not write into our buffers before this kernel is entered.
    bar = pltpu.get_barrier_semaphore()
    me = jax.lax.axis_index(axis_name)
    for k in range(n):
      pl.semaphore_signal(bar, device_id=_peer_id(axis_name, mesh_axes, jax.lax.rem(me + k, n)))
    pl.semaphore_wait(bar, n)
    for dma in descriptors(x_ref, bufs, sems):
      dma.start()

  def done_body(x_ref, *rest):
    ins = list(rest[: n - 1])            # aliased to the outputs; nothing loaded here
    sems = list(rest[2 * (n - 1):])
    for dma in descriptors(x_ref, ins, sems):
      dma.wait()   # sequential waits on the shared pair; each decrements its own bytes
    # Epoch fence (see scratch comment): no device leaves done_i before all finished done_i.
    exit_sem = sems[2 * slot + 2 + slot]
    me = jax.lax.axis_index(axis_name)
    for k in range(n):
      pl.semaphore_signal(
          exit_sem, device_id=_peer_id(axis_name, mesh_axes, jax.lax.rem(me + k, n))
      )
    pl.semaphore_wait(exit_sem, n)

  shapes = [shard_sds] * (n - 1)
  cp = _ag_cp(slot)
  start = pl.pallas_call(start_body, in_specs=[_HBM], out_specs=[_HBM] * (n - 1),
                         out_shape=shapes, scratch_shapes=scratch, compiler_params=cp)
  done = pl.pallas_call(done_body, in_specs=[_HBM] * n, out_specs=[_HBM] * (n - 1),
                        out_shape=shapes, scratch_shapes=scratch, compiler_params=cp,
                        input_output_aliases={i + 1: i for i in range(n - 1)})
  return start, done


def _sag_impl(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot=0):
  n = mesh.shape[axis_name]
  mesh_axes = tuple(mesh.axis_names)

  def body(xx):
    start, done = _make_ag_pair(n, jax.ShapeDtypeStruct(xx.shape, xx.dtype),
                                axis_name, mesh_axes, slot=slot)
    bufs = start(xx)
    got = done(xx, *bufs)                # <-- the layer's compute belongs in this gap
    # Position k holds shard (me-k) mod n; own shard at position 0. Permute to shard
    # order 0..n-1, then lay out exactly as lax.all_gather(tiled=True, axis=gather_axis).
    stacked = jnp.stack([xx] + list(got), axis=0)          # (n,) + shard
    me = jax.lax.axis_index(axis_name)
    order = jax.lax.rem(me - jnp.arange(n) + n, n)         # position of shard s
    permuted = jnp.take(stacked, order, axis=0)
    shard = xx.shape
    return jnp.moveaxis(permuted, 0, gather_axis).reshape(
        shard[:gather_axis] + (n * shard[gather_axis],) + shard[gather_axis + 1:]
    )

  return jax.shard_map(body, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec,
                       check_vma=False)(w)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6))
def split_all_gather(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot=0):
  """FSDP weight all-gather whose placement we own. Backward is XLA's reduce-scatter."""
  return _sag_impl(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot)


def _sag_fwd(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot=0):
  return _sag_impl(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot), None


def _sag_bwd(mesh, axis_name, gather_axis, in_spec, out_spec, slot, _res, ct):
  # The gather is LOGICALLY the identity (tiled all-gather of a tiled-sharded array), so the
  # logical cotangent IS the weight grad; a sharding constraint reshards it and lets GSPMD
  # fuse the pending partial-sum + slice into one reduce-scatter.
  #
  # Do NOT psum_scatter here. That transpose is correct only when the cotangent arrives as
  # UNSUMMED per-device partials, which is what another shard_map's transpose hands back
  # under check_vma=False (the `_make_cv_gather` context). Our consumer is the plain GSPMD
  # dot in DenseGeneral, whose autodiff delivers the already-summed logical cotangent --
  # psum_scatter on top of that over-counts by exactly n (CPU study: median ratio 8.000 at
  # n=8; identity ratio 1.000). The same defect explains the hoist's 0.174 lm_loss delta.
  g = jax.lax.with_sharding_constraint(ct, jax.sharding.NamedSharding(mesh, in_spec))
  return (g,)


split_all_gather.defvjp(_sag_fwd, _sag_bwd)
