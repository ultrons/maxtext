"""Bisection ladder from the known-good R3 up to the full split all-gather.

R1-R3 pass and R7 halts the core, and the gap between them contains five changes at once.
Each rung adds exactly ONE of them. The first halt kills the process, so the log says which
rung it reached -- one cluster cycle for the whole ladder instead of one per hypothesis.

  L1  one remote copy, DYNAMIC destination `o_ref.at[me]`        (vs R3's static dest)
  L2  + a second peer sharing the same ss/rs semaphore pair
  L3  + the barrier handshake (bare-int device_id, as R3 used)
  L4  + input_output_aliases on the DMA destination
  L5  + the full n-1 rotation

L1 also answers a question that costs nothing and would otherwise bite later: does a REMOTE
`o_ref.at[me]` resolve the slot with the SENDER's `me`? The code assumes device i writes
peer's slot i. If not, every device writes the same slot and the gather is silently wrong
even when it does not halt.
"""

import sys
import traceback

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
HBM = pl.BlockSpec(memory_space=pltpu.HBM)
DT = jnp.float32
SHARD = 256
SCRATCH = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
AX = "fsdp"

RESULTS = []


def gate(name, ok, detail=""):
  RESULTS.append((name, bool(ok), detail))
  print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}", flush=True)


def build(n, n_peers, use_barrier, alias, slots=True, local=True):
  """start/done pair with the rung's features enabled."""

  def descriptors(x_ref, o_ref, loc, ss, rs):
    me = jax.lax.axis_index(AX)
    dst = o_ref.at[me] if slots else o_ref
    if local:
      yield pltpu.make_async_copy(x_ref, dst, loc)
    for step in range(1, n_peers + 1):
      peer = jax.lax.rem(me + step, n)
      yield pltpu.make_async_remote_copy(x_ref, dst, ss, rs, device_id=peer)

  def start_body(x_ref, o_ref, loc, ss, rs):
    if use_barrier:
      bar = pltpu.get_barrier_semaphore()
      for d in range(n):
        pl.semaphore_signal(bar, device_id=d)   # bare int: R3 passed with this on a 1-D mesh
      pl.semaphore_wait(bar, n)
    for dma in descriptors(x_ref, o_ref, loc, ss, rs):
      dma.start()

  def done_body(x_ref, d_ref, o_ref, loc, ss, rs):
    for dma in descriptors(x_ref, d_ref, loc, ss, rs):
      dma.wait()

  cp = pltpu.CompilerParams(collective_id=7, allow_collective_id_without_custom_barrier=True,
                            has_side_effects=True)
  shape = jax.ShapeDtypeStruct((n, SHARD) if slots else (SHARD,), DT)
  start = pl.pallas_call(start_body, in_specs=[HBM], out_specs=HBM, out_shape=shape,
                         scratch_shapes=SCRATCH, compiler_params=cp)
  done_kw = dict(input_output_aliases={1: 0}) if alias else {}
  done = pl.pallas_call(done_body, in_specs=[HBM, HBM], out_specs=HBM, out_shape=shape,
                        scratch_shapes=SCRATCH, compiler_params=cp, **done_kw)
  return start, done


def run_rung(mesh, name, n_peers, use_barrier, alias, slots=True, local=True):
  n = mesh.shape[AX]
  start, done = build(n, n_peers, use_barrier, alias, slots, local)

  @jax.jit
  def f(w):
    return jax.shard_map(lambda xx: done(xx, start(xx)), mesh=mesh,
                         in_specs=(P(AX),),
                         out_specs=P(AX, None) if slots else P(AX), check_vma=False)(w)

  # 1-D so the per-device shard is (SHARD,), matching `o_ref.at[me]`. With a 2-D (n,SHARD)
  # array the shard is (1,SHARD) and the ranks differ:
  #   'tpu.enqueue_dma' op DMA source and target must have the same shape
  host = (np.arange(n)[:, None] * 1000.0 + np.arange(SHARD)[None, :]).astype(np.float32)
  w = jax.device_put(jnp.asarray(host.reshape(-1)), jax.sharding.NamedSharding(mesh, P(AX)))
  print(f"    {name}: executing ...", flush=True)
  raw = np.asarray(f(w))
  if not slots:
    # R3 semantics: device j receives sender (j-1)'s shard into its whole buffer.
    got = raw.reshape(n, SHARD)
    ok = all(np.array_equal(got[j], host[(j - 1) % n]) for j in range(n))
    gate(name, ok, "neighbour shard delivered" if ok else "wrong contents")
    return
  got = raw.reshape(n, n, SHARD)
  # Device j's slot s should hold device s's shard, for the senders that targeted j.
  senders = lambda j: [j] + [(j - k) % n for k in range(1, n_peers + 1)]
  ok = all(np.array_equal(got[j, s], host[s]) for j in range(n) for s in senders(j))
  bad = [(j, s) for j in range(n) for s in senders(j) if not np.array_equal(got[j, s], host[s])]
  gate(name, ok, "slot contents correct" if ok else f"wrong slots (sample {bad[:3]})")


def main():
  n = jax.device_count()
  mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(n), (AX,))
  print(f"jax {jax.__version__}  devices={n}", flush=True)

  # Each device j ends up with its own slot j (local copy) plus slots from senders that
  # targeted it. With n_peers=k, device j is written by senders j-1 .. j-k.
  # R3 is the known-good: barrier ON, one remote copy, STATIC whole-buffer destination,
  # no local copy. The previous ladder's L1 differed from it by THREE things at once
  # (dropped the barrier, added .at[me], added a local copy), which is why it told us
  # nothing. Each rung below changes exactly one.
  #        name                                 peers  barrier  alias  slots  local
  rungs = [
      ("L0 == R3 (static dest, barrier)",           1,   True,  False, False, False),
      ("L1 + slot dest o_ref.at[me]",               1,   True,  False, True,  False),
      ("L2 + local copy alongside remote",          1,   True,  False, True,  True),
      ("L3 + 2nd peer sharing ss/rs",               2,   True,  False, True,  True),
      ("L4 + input_output_aliases",                 2,   True,  True,  True,  True),
      ("L5 + full n-1 rotation",                n - 1,   True,  True,  True,  True),
  ]
  for name, k, bar, alias, slots, local in rungs:
    try:
      run_rung(mesh, name, k, bar, alias, slots, local)
    except Exception as e:  # noqa: BLE001
      gate(name, False, f"{type(e).__name__}: {str(e).strip().splitlines()[0][:400]}")
      print("  -- ladder stops at the first failure --", flush=True)
      break

  print("\n==== VERDICT ====", flush=True)
  for nm, ok, d in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {nm}  {d}", flush=True)
  sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
  main()
