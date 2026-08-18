"""All-gather built ONLY from static whole-buffer destinations.

The push model failed because the destination offset (`o_ref.at[me]`) is dynamic and has to be
interpreted in the receiver's buffer: the bytes silently never arrived. A pull model has the
same defect -- the offset still crosses the device boundary. But rung L0 proved that a remote
copy into a STATIC, whole-buffer destination delivers correctly.

So give every peer step its OWN output buffer and never index at all:

  at step k (k = 1..n-1, a PYTHON constant), device `me` pushes its shard into device
  (me+k)'s buffer k.

Buffer index k is static on both sides. Receiver j's buffer k therefore holds the shard of
device (j-k) mod n. The gathered array is then assembled OUTSIDE the kernel with ordinary XLA
ops, which is cheap and involves no DMA addressing.

Same start/done split: one `start` arms all n-1 sends, one `done` reconstructs and waits.
Both halves share one scratch signature (R5/R6: a mismatch hangs).
"""

import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
DT = jnp.float32
SHARD = 256
AX = "fsdp"
RESULTS = []


def gate(name, ok, detail=""):
  RESULTS.append((name, bool(ok), detail))
  print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}", flush=True)


def build(n):
  """start/done for the n-1 static-destination sends."""
  # One (send, recv) DMA semaphore pair per step: distinct buffers, distinct flags. Identical
  # in both halves.
  scratch = [pltpu.SemaphoreType.DMA] * (2 * (n - 1))

  def descriptors(x_ref, outs, sems):
    me = jax.lax.axis_index(AX)
    for i, k in enumerate(range(1, n)):
      peer = jax.lax.rem(me + k, n)
      yield pltpu.make_async_remote_copy(
          x_ref, outs[i], sems[2 * i], sems[2 * i + 1], device_id=peer
      )

  def start_body(x_ref, *rest):
    outs, sems = list(rest[: n - 1]), list(rest[n - 1:])
    bar = pltpu.get_barrier_semaphore()
    for d in range(n):
      pl.semaphore_signal(bar, device_id=d)
    pl.semaphore_wait(bar, n)
    for dma in descriptors(x_ref, outs, sems):
      dma.start()

  def done_body(x_ref, *rest):
    ins = list(rest[: n - 1])
    outs, sems = list(rest[n - 1: 2 * (n - 1)]), list(rest[2 * (n - 1):])
    del outs  # aliased to `ins`
    for dma in descriptors(x_ref, ins, sems):
      dma.wait()

  cp = pltpu.CompilerParams(collective_id=7, allow_collective_id_without_custom_barrier=True,
                            has_side_effects=True)
  shapes = [jax.ShapeDtypeStruct((SHARD,), DT)] * (n - 1)
  start = pl.pallas_call(start_body, in_specs=[HBM], out_specs=[HBM] * (n - 1),
                         out_shape=shapes, scratch_shapes=scratch, compiler_params=cp)
  done = pl.pallas_call(done_body, in_specs=[HBM] * n, out_specs=[HBM] * (n - 1),
                        out_shape=shapes, scratch_shapes=scratch, compiler_params=cp,
                        input_output_aliases={i + 1: i for i in range(n - 1)},
                        )
  return start, done


def main():
  n = jax.device_count()
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(n), (AX,))
  print(f"jax {jax.__version__}  devices={n}", flush=True)
  start, done = build(n)

  @jax.jit
  def f(w):
    def body(xx):
      bufs = start(xx)
      got = done(xx, *bufs)                     # <-- layer compute belongs in this gap
      return jnp.stack([xx] + list(got), axis=0)  # [own, step1, ..., step n-1]
    return jax.shard_map(body, mesh=mesh, in_specs=(P(AX),), out_specs=P(AX, None),
                         check_vma=False)(w)

  host = (np.arange(n)[:, None] * 1000.0 + np.arange(SHARD)[None, :]).astype(np.float32)
  w = jax.device_put(jnp.asarray(host.reshape(-1)),
                     jax.sharding.NamedSharding(mesh, P(AX)))
  print("    executing ...", flush=True)
  got = np.asarray(f(w)).reshape(n, n, SHARD)

  # device j, position 0 = own shard j; position k = shard (j-k) mod n
  ok = all(np.array_equal(got[j, k], host[(j - k) % n]) for j in range(n) for k in range(n))
  bad = [(j, k) for j in range(n) for k in range(n)
         if not np.array_equal(got[j, k], host[(j - k) % n])]
  gate("static-destination all-gather delivers every shard", ok,
       "all positions correct" if ok else f"wrong (sample {bad[:4]})")
  if not ok:
    for j in range(min(n, 2)):
      m = {k: next((t for t in range(n) if np.array_equal(got[j, k], host[t])), None)
           for k in range(n)}
      print(f"      dev{j} pos->shard {m}", flush=True)

  print("\n==== VERDICT ====", flush=True)
  for nm, o, d in RESULTS:
    print(f"  {'PASS' if o else 'FAIL'}  {nm}  {d}", flush=True)
  sys.exit(0 if all(o for _, o, _ in RESULTS) else 1)


if __name__ == "__main__":
  main()
