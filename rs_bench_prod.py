"""Isolated bench of the DS-v3 backward weight-grad reduce-scatter, at production mesh + shape.

Production op (siv-cn-ikqs2 profile, reduce-scatter.31):
  bf16[32, 2048, 7168] --psum_scatter over fsdp=128, scatter_dim=2--> bf16[32, 2048, 56]
  measured in-model: 3.374 ms/firing; bytes_accessed (in+out) = 946.9 MB => ~280 GB/s.

What this answers:
  1. WHICH SUBMESH the 128-way fsdp axis occupies. The per-dimension spread of the 128 devices'
     coordinates is what sets the bandwidth ceiling: one torus dim gives one link, three give three.
     Printed directly rather than inferred from a bandwidth that happens to divide nicely.
  2. Whether 280 GB/s is the WALL or a contention artifact. Here the RS runs with nothing else on
     the device, so its time IS the ceiling. Isolated ~= 3.37 ms => already at speed, and the only
     lever left is HIDING it (scheduling). Much faster isolated => the in-model op is contending.

Mesh must match production EXACTLY: MaxText ep-as-dp builds ['data','fsdp','fsdp_transpose','expert']
with dims (1, 128, 1, 8) and allow_split_physical_axes=FALSE. Axis order and split policy both decide
which physical torus dims each logical axis lands on.
"""

import functools
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

EP = int(os.environ.get("EP", 8))
FSDP = int(os.environ.get("FSDP", 128))
ITERS = int(os.environ.get("ITERS", 20))
E_LOCAL = int(os.environ.get("E_LOCAL", 32))  # experts per shard
K = int(os.environ.get("K", 2048))            # wo contracting dim
N = int(os.environ.get("N", 7168))            # wo output dim, sharded 128-way -> 56


def _coord(d):
  return tuple(getattr(d, "coords", ())) + (getattr(d, "core_on_chip", 0),)


def main():
  nd = jax.device_count()
  print(f"devices={nd} kind={jax.devices()[0].device_kind}", flush=True)
  assert nd >= EP * FSDP, f"need {EP*FSDP} devices, have {nd}"

  md = mesh_utils.create_device_mesh(
      (1, FSDP, 1, EP), jax.devices()[: EP * FSDP], allow_split_physical_axes=False
  )
  mesh = Mesh(md, ("data", "fsdp", "fsdp_transpose", "ep"))
  arr = np.asarray(md)

  # ---- SUBMESH REPORT -------------------------------------------------------
  fsdp_grp = arr[0, :, 0, 0]           # 128 devices along fsdp, other axes pinned
  ep_grp = arr[0, 0, 0, :]             # 8 devices along ep
  fc = [_coord(d) for d in fsdp_grp]
  ec = [_coord(d) for d in ep_grp]
  dims = len(fc[0])
  labels = ["x", "y", "z", "core"][:dims]
  print("\n=== SUBMESH ===", flush=True)
  spread = []
  for i in range(dims):
    vals = sorted({c[i] for c in fc})
    spread.append(len(vals))
    print(f"  fsdp dim {labels[i]}: {len(vals):3d} distinct {vals[:8]}{'...' if len(vals) > 8 else ''}", flush=True)
  print(f"  fsdp submesh shape  = {spread}  (product {np.prod(spread)}, axis size {FSDP})", flush=True)
  print(f"  physical dims spanned by fsdp = {sum(1 for s in spread if s > 1)}", flush=True)
  ep_spread = [len({c[i] for c in ec}) for i in range(dims)]
  print(f"  ep submesh shape    = {ep_spread}", flush=True)
  print(f"  fsdp first 6 coords = {fc[:6]}", flush=True)
  print(f"  ep coords           = {ec}", flush=True)

  # ---- BENCH ----------------------------------------------------------------
  in_bytes = E_LOCAL * K * N * 2
  out_bytes = E_LOCAL * K * (N // FSDP) * 2
  bytes_accessed = in_bytes + out_bytes
  ring_traffic = (FSDP - 1) * out_bytes

  in_sharding = NamedSharding(mesh, P("ep", None, None))

  # Build the global [EP*E_LOCAL, K, N] on device (7.5 GB global; 939 MB per device after the
  # ep shard, replicated across fsdp -- exactly the production per-device partial).
  @functools.partial(jax.jit, out_shardings=in_sharding)
  def _mk(key):
    return jax.random.normal(key, (EP * E_LOCAL, K, N), dtype=jnp.float32).astype(jnp.bfloat16)

  xg = _mk(jax.random.PRNGKey(0))
  jax.block_until_ready(xg)

  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(P("ep", None, None),),
      out_specs=P("ep", None, "fsdp"),
      check_rep=False,
  )
  def rs(x):
    return jax.lax.psum_scatter(x, "fsdp", scatter_dimension=2, tiled=True)

  fn = jax.jit(rs)
  out = fn(xg)
  jax.block_until_ready(out)
  print(f"\nout shape={out.shape} dtype={out.dtype}", flush=True)

  ts = []
  for _ in range(ITERS):
    t0 = time.perf_counter()
    r = fn(xg)
    jax.block_until_ready(r)
    ts.append((time.perf_counter() - t0) * 1e3)
  med, best = statistics.median(ts), min(ts)

  print(f"\n=== RESULT  mesh ep={EP} fsdp={FSDP}  shape=[{E_LOCAL},{K},{N}] bf16 ===", flush=True)
  print(f"  in/device       {in_bytes/1e6:9.1f} MB", flush=True)
  print(f"  out/device      {out_bytes/1e6:9.2f} MB", flush=True)
  print(f"  bytes_accessed  {bytes_accessed/1e6:9.1f} MB  (in+out)", flush=True)
  print(f"  ring traffic    {ring_traffic/1e6:9.1f} MB  ((P-1)*out)", flush=True)
  print(f"  wall median     {med:9.3f} ms   best {best:.3f} ms", flush=True)
  print(f"  BW(bytes_acc)   {bytes_accessed/1e6/med:9.1f} GB/s", flush=True)
  print(f"  BW(ring)        {ring_traffic/1e6/med:9.1f} GB/s", flush=True)
  print(f"  IN-MODEL REF    3.374 ms -> {bytes_accessed/1e6/3.374:.0f} GB/s", flush=True)
  print(f"  isolated/in-model = {med/3.374:.2f}x", flush=True)
  print("  NOTE: the wall figure above includes host dispatch+sync and is NOT comparable to the", flush=True)
  print("        profiler's DEVICE time. The trace below is the apples-to-apples number.", flush=True)

  # DEVICE time. Wall-vs-device is the whole reason the first read of this op went wrong; take the
  # measurement from the same instrument the in-model number came from.
  prof = os.environ.get("PROF_DIR", "")
  if prof:
    with jax.profiler.trace(prof):
      for _ in range(ITERS):
        r = fn(xg)
      jax.block_until_ready(r)
    print(f"  traced {ITERS} iters -> {prof}", flush=True)


if __name__ == "__main__":
  main()
