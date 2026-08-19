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


def _make_ag_pair(n, shard_sds, axis_name, mesh_axes, slot=0, n_steps=None, peer_fn=None):
  """Start/done pair for an n-way static-destination all-gather, CANONICAL form.

  Follows the Pallas Async Ops pattern (jax/tests/pallas/tpu_pallas_async_test.py):
  `start` RETURNS its DMA semaphores as outputs (`SemaphoreType.DMA(())` out_shape,
  `SEMAPHORE` memory-space out_spec) and `done` takes them as INPUTS. XLA keeps the sync
  flags alive and threads them through the dataflow, which retires the two failure
  classes of the reconstruct-in-done design at once: no deterministic-slot assumption
  (mismatched-scratch hang, R5) and no slot aliasing between co-scheduled pairs (the
  sag1d cluster halt, reproduced by R9 -- three padding attempts could not displace the
  slots because semaphore allocation is not positional-by-signature).

  `start` also aliases x through ({0: 0}), keeping the source buffer alive under the
  in-flight sends; `done` aliases every landing buffer to its outputs so downstream
  consumers order after the waits. Destinations stay per-step and STATIC (the
  dynamic-index refutation stands). The REGULAR exit fence in `done` serializes repeated
  executions, which reuse buffers -- and therefore sync-flag addresses -- under scan.
  """
  cp = _ag_cp(slot)
  sem = pltpu.SemaphoreType.DMA(())
  SEM = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)

  if n_steps is None:
    n_steps = n - 1
  if peer_fn is None:
    peer_fn = lambda me_, k: jax.lax.rem(me_ + k + 1, n)   # default: full rotation

  def descriptors(x_ref, buf, ss, rs):
    # ONE stacked (n_steps,)+shard landing buffer; buf.at[i] with a PYTHON-int i is a
    # STATIC destination (the refuted addressing was DYNAMIC me-indexing). peer_fn gives
    # the k-th destination rank on the gather axis; the bounded-fan-out form passes a
    # subgroup or group-stride rotation here.
    me = jax.lax.axis_index(axis_name)
    for i in range(n_steps):
      peer = peer_fn(me, i)
      yield pltpu.make_async_remote_copy(
          x_ref, buf.at[i], ss, rs, device_id=_peer_id(axis_name, mesh_axes, peer)
      )

  def start_body(x_ref, x_alias, buf, ss, rs):
    bar = pltpu.get_barrier_semaphore()
    me = jax.lax.axis_index(axis_name)
    for k in range(n):
      pl.semaphore_signal(bar, device_id=_peer_id(axis_name, mesh_axes, jax.lax.rem(me + k, n)))
    pl.semaphore_wait(bar, n)
    for dma in descriptors(x_ref, buf, ss, rs):
      dma.start()

  buf_sds = jax.ShapeDtypeStruct((n_steps,) + shard_sds.shape, shard_sds.dtype)
  start = pl.pallas_call(
      start_body,
      in_specs=[_HBM],
      out_shape=(shard_sds, buf_sds, sem, sem),
      out_specs=(_HBM, _HBM, SEM, SEM),
      input_output_aliases={0: 0},
      compiler_params=cp,
  )

  def done_body(x_ref, buf, ss, rs, _o, exit_sem):
    # _o is aliased to buf; nothing is loaded here.
    for _ in range(n_steps):
      pltpu.make_async_copy(x_ref, x_ref, ss).wait()      # one send completion each
    for i in range(n_steps):
      pltpu.make_async_copy(buf.at[i], buf.at[i], rs).wait()   # one arrival each
    # Epoch fence: scan reuses buffers, hence sync-flag addresses, across executions.
    me = jax.lax.axis_index(axis_name)
    for k in range(n):
      pl.semaphore_signal(exit_sem, device_id=_peer_id(axis_name, mesh_axes, jax.lax.rem(me + k, n)))
    pl.semaphore_wait(exit_sem, n)

  done = pl.pallas_call(
      done_body,
      in_specs=[_HBM, _HBM, SEM, SEM],
      out_specs=_HBM,
      out_shape=buf_sds,
      input_output_aliases={1: 0},
      scratch_shapes=[pltpu.SemaphoreType.REGULAR],
      compiler_params=cp,
  )
  return start, done


def _assemble(xx, got, order_idx, gather_axis, n_parts):
  """[own] + landed parts -> tiled all-gather layout along gather_axis."""
  stacked = jnp.concatenate([xx[None], got], axis=0)          # (n_parts,) + part
  permuted = jnp.take(stacked, order_idx, axis=0)
  shard = xx.shape
  return jnp.moveaxis(permuted, 0, gather_axis).reshape(
      shard[:gather_axis] + (n_parts * shard[gather_axis],) + shard[gather_axis + 1:]
  )


def _sag_impl(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot=0, group=0):
  if w.ndim < 2:
    # buf.at[i] on a landing buffer with a 1-D part squeezes to 1-D, which Mosaic
    # rejects ("All tiled squeezed dimensions must be of size 1"). Model weights are >=2-D.
    raise ValueError(f"split_all_gather requires a >=2-D shard, got shape {w.shape}")
  n = mesh.shape[axis_name]
  mesh_axes = tuple(mesh.axis_names)
  two_stage = 1 < group < n and n % group == 0
  s2 = n // group if two_stage else 1

  def body(xx):
    me = jax.lax.axis_index(axis_name)
    if not two_stage:
      start_k, done_k = _make_ag_pair(n, jax.ShapeDtypeStruct(xx.shape, xx.dtype),
                                      axis_name, mesh_axes, slot=slot)
      x_alias, buf, ss, rs = start_k(xx)
      got = done_k(x_alias, buf, ss, rs)   # <-- the layer's compute belongs in this gap
      order = jax.lax.rem(me - jnp.arange(n) + n, n)
      return _assemble(x_alias, got, order, gather_axis, n)

    # STAGE 1: gather within the contiguous subgroup of `group` ranks (fan-out group-1).
    # peer_fn must derive EVERYTHING from me_ (computed inside the kernel): a closure over
    # traced values raises "captures constants ... pass them as inputs".
    p1 = lambda me_, k: (me_ // group) * group + jax.lax.rem(me_ - (me_ // group) * group + k + 1, group)
    st1, dn1 = _make_ag_pair(n, jax.ShapeDtypeStruct(xx.shape, xx.dtype), axis_name,
                             mesh_axes, slot=slot, n_steps=group - 1, peer_fn=p1)
    xa1, b1, ss1, rs1 = st1(xx)
    g1 = dn1(xa1, b1, ss1, rs1)
    order1 = jax.lax.rem(off - jnp.arange(group) + group, group)
    block = _assemble(xa1, g1, order1, gather_axis, group)     # this subgroup's block

    # STAGE 2: exchange assembled blocks across the s2 supergroups at stride `group`
    # (fan-out s2-1). Depends on stage 1's done through `block`.
    p2 = lambda me_, k: jax.lax.rem(me_ // group + k + 1, s2) * group + jax.lax.rem(me_, group)
    st2, dn2 = _make_ag_pair(n, jax.ShapeDtypeStruct(block.shape, block.dtype), axis_name,
                             mesh_axes, slot=slot + 3, n_steps=s2 - 1, peer_fn=p2)
    xa2, b2, ss2, rs2 = st2(block)
    g2 = dn2(xa2, b2, ss2, rs2)
    order2 = jax.lax.rem(gi - jnp.arange(s2) + s2, s2)
    return _assemble(xa2, g2, order2, gather_axis, s2)

  return jax.shard_map(body, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec,
                       check_vma=False)(w)


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


def _make_ag_pair(n, shard_sds, axis_name, mesh_axes, slot=0, n_steps=None, peer_fn=None):
  """Start/done pair for an n-way static-destination all-gather, CANONICAL form.

  Follows the Pallas Async Ops pattern (jax/tests/pallas/tpu_pallas_async_test.py):
  `start` RETURNS its DMA semaphores as outputs (`SemaphoreType.DMA(())` out_shape,
  `SEMAPHORE` memory-space out_spec) and `done` takes them as INPUTS. XLA keeps the sync
  flags alive and threads them through the dataflow, which retires the two failure
  classes of the reconstruct-in-done design at once: no deterministic-slot assumption
  (mismatched-scratch hang, R5) and no slot aliasing between co-scheduled pairs (the
  sag1d cluster halt, reproduced by R9 -- three padding attempts could not displace the
  slots because semaphore allocation is not positional-by-signature).

  `start` also aliases x through ({0: 0}), keeping the source buffer alive under the
  in-flight sends; `done` aliases every landing buffer to its outputs so downstream
  consumers order after the waits. Destinations stay per-step and STATIC (the
  dynamic-index refutation stands). The REGULAR exit fence in `done` serializes repeated
  executions, which reuse buffers -- and therefore sync-flag addresses -- under scan.
  """
  cp = _ag_cp(slot)
  sem = pltpu.SemaphoreType.DMA(())
  SEM = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)

  if n_steps is None:
    n_steps = n - 1
  if peer_fn is None:
    peer_fn = lambda me_, k: jax.lax.rem(me_ + k + 1, n)   # default: full rotation

  def descriptors(x_ref, buf, ss, rs):
    # ONE stacked (n_steps,)+shard landing buffer; buf.at[i] with a PYTHON-int i is a
    # STATIC destination (the refuted addressing was DYNAMIC me-indexing). peer_fn gives
    # the k-th destination rank on the gather axis; the bounded-fan-out form passes a
    # subgroup or group-stride rotation here.
    me = jax.lax.axis_index(axis_name)
    for i in range(n_steps):
      peer = peer_fn(me, i)
      yield pltpu.make_async_remote_copy(
          x_ref, buf.at[i], ss, rs, device_id=_peer_id(axis_name, mesh_axes, peer)
      )

  def start_body(x_ref, x_alias, buf, ss, rs):
    bar = pltpu.get_barrier_semaphore()
    me = jax.lax.axis_index(axis_name)
    for k in range(n):
      pl.semaphore_signal(bar, device_id=_peer_id(axis_name, mesh_axes, jax.lax.rem(me + k, n)))
    pl.semaphore_wait(bar, n)
    for dma in descriptors(x_ref, buf, ss, rs):
      dma.start()

  buf_sds = jax.ShapeDtypeStruct((n_steps,) + shard_sds.shape, shard_sds.dtype)
  start = pl.pallas_call(
      start_body,
      in_specs=[_HBM],
      out_shape=(shard_sds, buf_sds, sem, sem),
      out_specs=(_HBM, _HBM, SEM, SEM),
      input_output_aliases={0: 0},
      compiler_params=cp,
  )

  def done_body(x_ref, buf, ss, rs, _o, exit_sem):
    # _o is aliased to buf; nothing is loaded here.
    for _ in range(n_steps):
      pltpu.make_async_copy(x_ref, x_ref, ss).wait()      # one send completion each
    for i in range(n_steps):
      pltpu.make_async_copy(buf.at[i], buf.at[i], rs).wait()   # one arrival each
    # Epoch fence: scan reuses buffers, hence sync-flag addresses, across executions.
    me = jax.lax.axis_index(axis_name)
    for k in range(n):
      pl.semaphore_signal(exit_sem, device_id=_peer_id(axis_name, mesh_axes, jax.lax.rem(me + k, n)))
    pl.semaphore_wait(exit_sem, n)

  done = pl.pallas_call(
      done_body,
      in_specs=[_HBM, _HBM, SEM, SEM],
      out_specs=_HBM,
      out_shape=buf_sds,
      input_output_aliases={1: 0},
      scratch_shapes=[pltpu.SemaphoreType.REGULAR],
      compiler_params=cp,
  )
  return start, done


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def split_all_gather(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot=0, group=0):
  """FSDP weight all-gather whose placement we own. Backward is XLA's reduce-scatter.

  group > 1 selects the TWO-STAGE bounded-fan-out form: stage 1 gathers within
  contiguous subgroups of `group` ranks, stage 2 exchanges assembled blocks across the
  n/group supergroups. Per-kernel peer fan-out drops from n-1 to max(group, n/group)-1.
  Motivation: sag1g halts at 512 chips with 127 DISTINCT peers while every single-core
  width behaviour (127-deep fan-out, R12) passes on the rig, so the remaining suspects
  are per-peer hardware state; bounding fan-out sidesteps them and is the shape larger
  meshes need anyway. Cost: stage 2 depends on stage 1's done, so the cross-group part
  of the transfer has roughly half the hiding window.
  """
  return _sag_impl(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot, group)


def _sag_fwd(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot=0, group=0):
  return _sag_impl(w, mesh, axis_name, gather_axis, in_spec, out_spec, slot, group), None


def _sag_bwd(mesh, axis_name, gather_axis, in_spec, out_spec, slot, group, _res, ct):
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
