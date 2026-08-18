"""RUNTIME gate for the start/done split-phase collective pattern (v7x, 2x2x1 = 8 devices).

The static Mosaic gate already passed. This answers what it cannot:

  R1  does `done`'s RECONSTRUCTED descriptor actually wait on the DMA that `start` armed?
      (local HBM->HBM, correctness assert -- this is the binary question)
  R2  does XLA keep the ordering, i.e. is the result still correct when independent
      compute sits between start and done?
  R3  remote cross-device start/done over the fsdp axis, correctness assert
  R4  ADJACENCY: does a matmul running next to an in-flight start/done keep its
      throughput? `mpmd-map-sc-barrier-blocks-fsdp-ag` records an SC kernel next to a
      gmm poisoning an all-gather 47 -> 3 GB/s. If we reproduce that here, the pattern
      has the old pathology in new syntax and the answer is no.

Exit code is nonzero if any gate fails, so the pod's status carries the verdict.
"""

import functools
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
DT = jnp.float32
N = 8192          # per-device element count for the DMA
_SEM = [pltpu.SemaphoreType.DMA]

RESULTS = []


def gate(name, ok, detail=""):
  RESULTS.append((name, bool(ok), detail))
  print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}", flush=True)


# ---------------------------------------------------------------------------
# local start / done.  No semaphore is passed: each kernel allocates its own DMA
# semaphore as scratch, and `done` rebuilds the identical descriptor.
# ---------------------------------------------------------------------------
def _start_body(x_ref, o_ref, sem):
  pltpu.make_async_copy(x_ref, o_ref, sem).start()


def _done_body(x_ref, d_ref, o_ref, sem):
  pltpu.make_async_copy(x_ref, d_ref, sem).wait()


_start = pl.pallas_call(_start_body, in_specs=[HBM], out_specs=HBM,
                        out_shape=jax.ShapeDtypeStruct((N,), DT), scratch_shapes=_SEM)
_done = pl.pallas_call(_done_body, in_specs=[HBM, HBM], out_specs=HBM,
                       out_shape=jax.ShapeDtypeStruct((N,), DT), scratch_shapes=_SEM,
                       input_output_aliases={1: 0})


@jax.jit
def local_startdone(x):
  return _done(x, _start(x))


@jax.jit
def local_startdone_with_compute(x, w):
  d = _start(x)
  busy = w @ w              # independent: XLA is free to put this in the gap
  return _done(x, d), busy.sum()


def r1_r2(dev):
  x = jax.device_put(jnp.arange(N, dtype=DT) * 0.001 + 1.0, dev)
  got = np.asarray(local_startdone(x))
  gate("R1 local start/done delivers the bytes",
       np.allclose(got, np.asarray(x)),
       f"max|err|={np.max(np.abs(got - np.asarray(x))):.3e}")

  w = jax.device_put(jnp.ones((512, 512), DT), dev)
  got2, _ = local_startdone_with_compute(x, w)
  got2 = np.asarray(got2)
  gate("R2 ordering holds with compute between start and done",
       np.allclose(got2, np.asarray(x)),
       f"max|err|={np.max(np.abs(got2 - np.asarray(x))):.3e}")


# ---------------------------------------------------------------------------
# remote start / done across the fsdp axis: device i sends its shard to device i+1.
# A barrier semaphore handshake makes sure the receiver's buffer exists before any
# sender writes into it; without it the remote write races the allocation.
# ---------------------------------------------------------------------------
def _rstart_body(x_ref, o_ref, ss, rs):
  n = jax.lax.axis_size("fsdp")
  me = jax.lax.axis_index("fsdp")
  bar = pltpu.get_barrier_semaphore()
  for d in range(8):
    pltpu.semaphore_signal(bar, device_id=d)
  pltpu.semaphore_wait(bar, n)
  pltpu.make_async_remote_copy(x_ref, o_ref, ss, rs,
                               device_id=jax.lax.rem(me + 1, n)).start()


def _rdone_body(x_ref, d_ref, o_ref, ss, rs):
  n = jax.lax.axis_size("fsdp")
  me = jax.lax.axis_index("fsdp")
  pltpu.make_async_remote_copy(x_ref, d_ref, ss, rs,
                               device_id=jax.lax.rem(me + 1, n)).wait()


def r3(mesh):
  sems = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
  cp = pltpu.CompilerParams(collective_id=7)
  rstart = pl.pallas_call(_rstart_body, in_specs=[HBM], out_specs=HBM,
                          out_shape=jax.ShapeDtypeStruct((N,), DT),
                          scratch_shapes=sems, compiler_params=cp)
  rdone = pl.pallas_call(_rdone_body, in_specs=[HBM, HBM], out_specs=HBM,
                         out_shape=jax.ShapeDtypeStruct((N,), DT),
                         scratch_shapes=sems, compiler_params=cp,
                         input_output_aliases={1: 0})

  @jax.jit
  def f(x):
    return jax.shard_map(lambda xx: rdone(xx, rstart(xx)), mesh=mesh,
                         in_specs=(P("fsdp"),), out_specs=P("fsdp"), check_vma=False)(x)

  nd = mesh.shape["fsdp"]
  host = (np.arange(nd)[:, None] * 1000.0 + np.arange(N)[None, :] * 0.001).astype(np.float32)
  x = jax.device_put(jnp.asarray(host.reshape(-1)),
                     jax.sharding.NamedSharding(mesh, P("fsdp")))
  got = np.asarray(f(x)).reshape(nd, N)
  # device i receives from device i-1
  want = host[(np.arange(nd) - 1) % nd]
  gate("R3 remote start/done delivers the neighbour's shard",
       np.allclose(got, want), f"max|err|={np.max(np.abs(got - want)):.3e}")


# ---------------------------------------------------------------------------
# R4 adjacency: matmul throughput alone vs alongside an in-flight start/done.
# ---------------------------------------------------------------------------
def r4(dev):
  M = 2048
  w = jax.device_put(jnp.ones((M, M), DT), dev)
  x = jax.device_put(jnp.arange(N, dtype=DT), dev)

  @jax.jit
  def mm_only(w):
    a = w
    for _ in range(8):
      a = a @ w
    return a.sum()

  @jax.jit
  def mm_next_to_dma(x, w):
    d = _start(x)
    a = w
    for _ in range(8):
      a = a @ w
    return _done(x, d), a.sum()

  def bench(fn, *a):
    jax.block_until_ready(fn(*a))
    t = time.perf_counter()
    for _ in range(20):
      r = fn(*a)
    jax.block_until_ready(r)
    return (time.perf_counter() - t) / 20

  t_alone = bench(mm_only, w)
  t_adj = bench(mm_next_to_dma, x, w)
  flops = 8 * 2 * M**3
  g_alone, g_adj = flops / t_alone / 1e12, flops / t_adj / 1e12
  ratio = g_adj / g_alone
  gate("R4 matmul keeps throughput next to an in-flight DMA",
       ratio > 0.90,
       f"{g_alone:.1f} -> {g_adj:.1f} TFLOP/s ({ratio*100:.1f}% retained)")


if __name__ == "__main__":
  print(f"jax {jax.__version__}  devices={jax.device_count()}", flush=True)
  dev = jax.devices()[0]
  nd = jax.device_count()
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(nd), ("fsdp",))

  for fn, args in ((r1_r2, (dev,)), (r3, (mesh,)), (r4, (dev,))):
    try:
      fn(*args)
    except Exception as e:  # noqa: BLE001
      gate(fn.__name__, False, f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}")

  print("\n==== VERDICT ====", flush=True)
  for n, ok, d in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {n}  {d}", flush=True)
  sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)
