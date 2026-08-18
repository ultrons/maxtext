"""Feasibility gate for the start/done split-phase collective pattern.

The whole design rests on ONE binary question: can a DMA started inside kernel A be
waited on inside a SEPARATE kernel B? That requires the DMA semaphore to outlive a
single pallas_call. Pallas exposes SEMAPHORE as a first-class MemorySpace, which is
what makes this plausible, but scratch semaphores are kernel-scoped.

Probe order, cheapest first. Each stage prints PASS/FAIL and why, so a failure tells us
which layer refused rather than just "it broke".

  P1  can a pallas_call take an operand in SEMAPHORE memory space at all (trace only)
  P2  does a two-kernel start/done LOWER through Mosaic (AOT, no hardware)
  P3  (hardware only, not run here) do the bytes actually arrive

Run:  python3 startdone_probe.py
"""

import traceback

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

N = 512
DT = jnp.float32


def _report(tag, fn):
  try:
    fn()
    print(f"  PASS  {tag}")
    return True
  except Exception as e:  # noqa: BLE001 - we want the class + first line
    first = str(e).strip().split("\n")[0][:220]
    print(f"  FAIL  {tag}\n          {type(e).__name__}: {first}")
    return False


# --------------------------------------------------------------------------------
# P1: is SEMAPHORE usable as a pallas_call operand memory space?
# --------------------------------------------------------------------------------
def p1_semaphore_as_operand():
  def body(x_ref, sem_ref, o_ref):
    pltpu.make_async_copy(x_ref, o_ref, sem_ref).start()
    pltpu.make_async_copy(x_ref, o_ref, sem_ref).wait()

  f = pl.pallas_call(
      body,
      in_specs=[
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
      ],
      out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
      out_shape=jax.ShapeDtypeStruct((N,), DT),
  )
  x = jax.ShapeDtypeStruct((N,), DT)
  sem = jax.ShapeDtypeStruct(pltpu.SemaphoreType.DMA(()).shape,
                             pltpu.SemaphoreType.DMA(()).dtype)
  jax.eval_shape(f, x, sem)


# --------------------------------------------------------------------------------
# P2: two SEPARATE kernels sharing one semaphore -- the actual pattern.
#     `start` issues the DMA and returns without waiting. `done` reconstructs the
#     identical descriptor and waits. The destination buffer is threaded through as
#     `done`'s input so XLA has a REAL data dependency and cannot hoist done above
#     start (a semaphore alone is invisible to the scheduler).
# --------------------------------------------------------------------------------
def _start_body(x_ref, sem_ref, o_ref):
  pltpu.make_async_copy(x_ref, o_ref, sem_ref).start()


def _done_body(x_ref, d_ref, sem_ref, o_ref):
  # ONLY wait. Reconstruct the identical descriptor so the wait knows the byte count;
  # no load, because Mosaic forbids loads from an HBM ref ("Loads are only allowed on
  # VMEM and SMEM references"). The result is delivered by aliasing d_ref to the output.
  pltpu.make_async_copy(x_ref, d_ref, sem_ref).wait()


def p2_two_kernel_start_done():
  sem_sds = jax.ShapeDtypeStruct(pltpu.SemaphoreType.DMA(()).shape,
                                 pltpu.SemaphoreType.DMA(()).dtype)
  any_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  sem_spec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)

  start = pl.pallas_call(
      _start_body,
      in_specs=[any_spec, sem_spec],
      out_specs=any_spec,
      out_shape=jax.ShapeDtypeStruct((N,), DT),
      input_output_aliases={},
  )
  done = pl.pallas_call(
      _done_body,
      in_specs=[any_spec, any_spec, sem_spec],
      out_specs=any_spec,
      out_shape=jax.ShapeDtypeStruct((N,), DT),
      input_output_aliases={1: 0},   # d_ref IS the output; no copy, no HBM load
  )

  def f(x, sem):
    d = start(x, sem)          # DMA in flight; kernel body already exited
    # ... other work would be scheduled here ...
    return done(x, d, sem)     # real data dep on d, so done cannot float above start

  jax.eval_shape(f, jax.ShapeDtypeStruct((N,), DT), sem_sds)


def p2_mosaic_lowering():
  """Same as P2 but forced all the way through Mosaic on a virtual TPU (no hardware)."""
  from jax.experimental import topologies

  topo = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu")
  sem_sds = jax.ShapeDtypeStruct(pltpu.SemaphoreType.DMA(()).shape,
                                 pltpu.SemaphoreType.DMA(()).dtype)
  any_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  sem_spec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)

  start = pl.pallas_call(_start_body, in_specs=[any_spec, sem_spec], out_specs=any_spec,
                         out_shape=jax.ShapeDtypeStruct((N,), DT))
  done = pl.pallas_call(_done_body, in_specs=[any_spec, any_spec, sem_spec],
                        out_specs=any_spec, out_shape=jax.ShapeDtypeStruct((N,), DT),
                        input_output_aliases={1: 0})

  def f(x, sem):
    return done(x, start(x, sem), sem)

  with jax.default_device(topo.devices[0]):
    jax.jit(f).lower(jax.ShapeDtypeStruct((N,), DT), sem_sds).compile()


if __name__ == "__main__":
  print(f"jax {jax.__version__}")
  print("P1  SEMAPHORE as a pallas_call operand")
  ok1 = _report("trace", p1_semaphore_as_operand)
  print("P2  two-kernel start/done sharing one semaphore")
  ok2 = _report("trace", p2_two_kernel_start_done)
  ok3 = _report("mosaic lowering (virtual tpu7x)", p2_mosaic_lowering) if ok2 else False
  print()
  print(f"VERDICT: operand={ok1} two_kernel_trace={ok2} mosaic={ok3}")
  if not (ok1 and ok2):
    print("=> cross-kernel semaphore lifetime is NOT expressible as written;")
    print("   the start/done split collapses to a single kernel unless another")
    print("   mechanism carries the semaphore.")
