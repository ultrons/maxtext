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
  cp = pltpu.CompilerParams(collective_id=7, allow_collective_id_without_custom_barrier=True,
                            has_side_effects=True)
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
# R5 MISMATCHED SCRATCH -- the failure mode that would bite silently in the model.
# R1 only proves the two kernels' scratch semaphores coincide when their scratch
# FOOTPRINTS are identical. A real gather gives `start` and `done` different scratch
# (done needs VMEM staging, start does not). If Mosaic then assigns the DMA semaphore a
# different slot, the wait targets the wrong sync flag and it fails as a hang or as
# corruption, never as an error. So: give `done` extra scratch AHEAD of its semaphore
# and check the bytes still arrive.
# ---------------------------------------------------------------------------
def _done_body_mm(x_ref, d_ref, o_ref, pad_vmem, pad_sem, sem):
  pltpu.make_async_copy(x_ref, d_ref, sem).wait()


def r5(dev):
  done_mm = pl.pallas_call(
      _done_body_mm, in_specs=[HBM, HBM], out_specs=HBM,
      out_shape=jax.ShapeDtypeStruct((N,), DT),
      scratch_shapes=[pltpu.VMEM((256, 128), DT), pltpu.SemaphoreType.REGULAR,
                      pltpu.SemaphoreType.DMA],
      input_output_aliases={1: 0})

  @jax.jit
  def f(x):
    return done_mm(x, _start(x))

  x = jax.device_put(jnp.arange(N, dtype=DT) * 0.001 + 3.0, dev)
  print("    r5: lowering/compiling ...", flush=True)
  c = jax.jit(f).lower(x).compile()
  print("    r5: COMPILED ok; executing (a hang past here == wrong sync flag) ...", flush=True)
  got = np.asarray(c(x))
  print("    r5: executed", flush=True)
  gate("R5 start/done survives MISMATCHED scratch footprints",
       np.allclose(got, np.asarray(x)),
       f"max|err|={np.max(np.abs(got - np.asarray(x))):.3e}")


# ---------------------------------------------------------------------------
# R6 THE FIX: mismatched scratch hangs (R5), so make `start` declare the SAME scratch
# signature as `done` even though it does not use the padding. If this passes, the rule
# for every start/done pair we build is: byte-identical scratch_shapes on both halves.
# ---------------------------------------------------------------------------
_PAD = [pltpu.VMEM((256, 128), DT), pltpu.SemaphoreType.REGULAR, pltpu.SemaphoreType.DMA]


def _start_body_pad(x_ref, o_ref, pad_vmem, pad_sem, sem):
  pltpu.make_async_copy(x_ref, o_ref, sem).start()


def r6(dev):
  start_pad = pl.pallas_call(_start_body_pad, in_specs=[HBM], out_specs=HBM,
                             out_shape=jax.ShapeDtypeStruct((N,), DT), scratch_shapes=_PAD)
  done_pad = pl.pallas_call(_done_body_mm, in_specs=[HBM, HBM], out_specs=HBM,
                            out_shape=jax.ShapeDtypeStruct((N,), DT), scratch_shapes=_PAD,
                            input_output_aliases={1: 0})

  def f(x):
    return done_pad(x, start_pad(x))

  x = jax.device_put(jnp.arange(N, dtype=DT) * 0.001 + 5.0, dev)
  c = jax.jit(f).lower(x).compile()
  print("    r6: COMPILED ok; executing ...", flush=True)
  got = np.asarray(c(x))
  gate("R6 MATCHED scratch on both halves restores correctness",
       np.allclose(got, np.asarray(x)),
       f"max|err|={np.max(np.abs(got - np.asarray(x))):.3e}")


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



# ---------------------------------------------------------------------------
# R7  the REAL forward split all-gather: does it match lax.all_gather exactly, and does
#     its VJP match the stock gather's VJP? A wrong transpose corrupts weight grads
#     silently, so the gradient is checked, not just the forward value.
# ---------------------------------------------------------------------------
def r7(mesh):
  """Numerics gate for the STATIC-destination split_all_gather, both gather axes.

  R7a/c: forward bit-identical to lax.all_gather(tiled=True). R7b/d: gradient bit-identical
  to the stock gather's gradient (a wrong psum_scatter transpose corrupts weight grads
  silently, so the gradient is the check that matters). Shapes are the real shared-expert
  kernel sharded over this rig's fsdp=8: wi (7168,2048) on axis 0, wo (2048,7168) on axis 1.
  """
  from maxtext.kernels.startdone import split_all_gather

  ax = "fsdp"
  rng = np.random.default_rng(0)

  def check(tag, full_shape, g_axis, in_spec):
    out_spec = P(None, None)

    def ours(w):
      return split_all_gather(w, mesh, ax, g_axis, in_spec, out_spec)

    def stock(w):
      return jax.shard_map(
          lambda x: jax.lax.all_gather(x, ax, axis=g_axis, tiled=True),
          mesh=mesh, in_specs=(in_spec,), out_specs=out_spec, check_vma=False)(w)

    host = rng.standard_normal(full_shape).astype(np.float32)
    w = jax.device_put(jnp.asarray(host), jax.sharding.NamedSharding(mesh, in_spec))
    go, gs = np.asarray(jax.jit(ours)(w)), np.asarray(jax.jit(stock)(w))
    gate(f"R7{tag}1 split AG matches lax.all_gather (axis {g_axis})",
         np.array_equal(go, gs), f"max|err|={np.max(np.abs(go - gs)):.3e}")

    co = jax.device_put(jnp.asarray(rng.standard_normal(full_shape).astype(np.float32)),
                        jax.sharding.NamedSharding(mesh, out_spec))
    loss = lambda f: (lambda w: jnp.sum(f(w) * co))
    do = np.asarray(jax.jit(jax.grad(loss(ours)))(w))
    ds = np.asarray(jax.jit(jax.grad(loss(stock)))(w))
    gate(f"R7{tag}2 split AG VJP matches stock VJP (axis {g_axis})",
         np.array_equal(do, ds), f"max|err|={np.max(np.abs(do - ds)):.3e}")

  check("a", (7168, 2048), 0, P(ax, None))   # wi: embed-sharded on axis 0
  check("b", (2048, 7168), 1, P(None, ax))   # wo: embed-sharded on axis 1



if __name__ == "__main__":
  print(f"jax {jax.__version__}  devices={jax.device_count()}", flush=True)
  dev = jax.devices()[0]
  nd = jax.device_count()
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(nd), ("fsdp",))

  import os
  sel = os.environ.get("GATES", "")
  allg = {"r1_r2": (r1_r2, (dev,)), "r3": (r3, (mesh,)), "r5": (r5, (dev,)), "r6": (r6, (dev,)), "r7": (r7, (mesh,)), "r4": (r4, (dev,))}
  chosen = [allg[k] for k in (sel.split(",") if sel else allg) if k in allg]
  for fn, args in chosen:
    try:
      fn(*args)
    except Exception as e:  # noqa: BLE001
      gate(fn.__name__, False, f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}")

  print("\n==== VERDICT ====", flush=True)
  for n, ok, d in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {n}  {d}", flush=True)
  sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)
