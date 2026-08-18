"""Feasibility gate for the start/done split-phase collective pattern.

The pattern: split a collective into two kernels. `start` issues the DMAs and returns
WITHOUT waiting, so XLA can schedule real compute after it. `done` RECONSTRUCTS the
identical DMA descriptor and waits on it. Both look like ordinary TC custom-calls to
XLA, so it schedules around them instead of fencing at them.

Crucially NO semaphore is passed between the kernels. Each kernel allocates its own DMA
semaphore as kernel scratch, which is what puts it in sync-flag memory; because the
allocation is deterministic, `done`'s reconstructed descriptor names the same physical
sync flag that `start` armed.

(An earlier version of this probe threaded the semaphore through as a pallas_call
operand. That is wrong and does not lower: an operand semaphore is allocated in VMEM and
Mosaic rejects the wait with
  LLO_CHECK ... sync_flag->memory_space() == kSflag || kBarnaCoreSflag  vmem
Setting memory_space=pltpu.SEMAPHORE on the input BlockSpec does not change it.)

  P1  local HBM->HBM start/done, two kernels, Mosaic lowering on a virtual tpu7x
  P2  same, with real compute between start and done (the point of the pattern)
  P3  remote (cross-device) start/done -- the shape a weight all-gather actually needs
  P4  (hardware) do the bytes arrive, and does the neighbouring gmm keep its bandwidth

Run:  python3 startdone_probe.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

N = 512
DT = jnp.float32
HBM = pl.BlockSpec(memory_space=pltpu.HBM)


def _report(tag, fn):
  try:
    fn()
    print(f"  PASS  {tag}")
    return True
  except Exception as e:  # noqa: BLE001
    first = str(e).strip().split("\n")[0][:240]
    print(f"  FAIL  {tag}\n          {type(e).__name__}: {first}")
    return False


def _topo_dev():
  from jax.experimental import topologies

  return topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices[0]


# ---------------------------------------------------------------------------
# start: arm the DMA, do NOT wait.   done: rebuild the same descriptor, wait.
# ---------------------------------------------------------------------------
def _start_body(x_ref, o_ref, sem):
  pltpu.make_async_copy(x_ref, o_ref, sem).start()


def _done_body(x_ref, d_ref, o_ref, sem):
  # d_ref is aliased to the output, so no HBM load is needed (Mosaic forbids those).
  pltpu.make_async_copy(x_ref, d_ref, sem).wait()


_SEM = [pltpu.SemaphoreType.DMA]

start = pl.pallas_call(
    _start_body,
    in_specs=[HBM],
    out_specs=HBM,
    out_shape=jax.ShapeDtypeStruct((N,), DT),
    scratch_shapes=_SEM,
)
done = pl.pallas_call(
    _done_body,
    in_specs=[HBM, HBM],
    out_specs=HBM,
    out_shape=jax.ShapeDtypeStruct((N,), DT),
    scratch_shapes=_SEM,
    input_output_aliases={1: 0},  # the DMA destination IS the output
)


def p1_local():
  def f(x):
    d = start(x)        # in flight; kernel body has already exited
    return done(x, d)   # real data dep on d, so done cannot float above start

  with jax.default_device(_topo_dev()):
    jax.jit(f).lower(jax.ShapeDtypeStruct((N,), DT)).compile()


def p2_compute_between():
  """The whole point: independent compute scheduled between start and done."""

  def f(x, w):
    d = start(x)
    busy = (w @ w).sum()            # no dependency on d -- XLA may place it in the gap
    return done(x, d), busy

  with jax.default_device(_topo_dev()):
    jax.jit(f).lower(
        jax.ShapeDtypeStruct((N,), DT), jax.ShapeDtypeStruct((256, 256), DT)
    ).compile()


def p3_remote():
  """Cross-device start/done -- the shape a weight all-gather actually needs."""
  from jax.experimental import topologies

  topo = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
  mesh = jax.sharding.Mesh(topo.devices[:8], ("fsdp",))
  P = jax.sharding.PartitionSpec

  def _rstart_body(x_ref, o_ref, ss, rs):
    nxt = jax.lax.rem(jax.lax.axis_index("fsdp") + 1, jax.lax.axis_size("fsdp"))
    pltpu.make_async_remote_copy(x_ref, o_ref, ss, rs, device_id=nxt).start()

  def _rdone_body(x_ref, d_ref, o_ref, ss, rs):
    nxt = jax.lax.rem(jax.lax.axis_index("fsdp") + 1, jax.lax.axis_size("fsdp"))
    pltpu.make_async_remote_copy(x_ref, d_ref, ss, rs, device_id=nxt).wait()

  sems = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
  rstart = pl.pallas_call(_rstart_body, in_specs=[HBM], out_specs=HBM,
                          out_shape=jax.ShapeDtypeStruct((N,), DT), scratch_shapes=sems)
  rdone = pl.pallas_call(_rdone_body, in_specs=[HBM, HBM], out_specs=HBM,
                         out_shape=jax.ShapeDtypeStruct((N,), DT), scratch_shapes=sems,
                         input_output_aliases={1: 0})

  def f(x):
    return jax.shard_map(
        lambda xx: rdone(xx, rstart(xx)),
        mesh=mesh, in_specs=(P("fsdp"),), out_specs=P("fsdp"), check_vma=False,
    )(x)

  with jax.default_device(topo.devices[0]):
    jax.jit(f).lower(jax.ShapeDtypeStruct((N * 8,), DT)).compile()


if __name__ == "__main__":
  print(f"jax {jax.__version__}")
  print("P1  local HBM->HBM start/done, semaphore reconstructed not passed")
  ok1 = _report("mosaic lowering", p1_local)
  print("P2  independent compute between start and done")
  ok2 = _report("mosaic lowering", p2_compute_between)
  print("P3  remote (cross-device) start/done")
  ok3 = _report("mosaic lowering", p3_remote)
  print()
  print(f"VERDICT: local={ok1} compute_between={ok2} remote={ok3}")
