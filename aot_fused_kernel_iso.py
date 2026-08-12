"""Isolated AOT: does the VENDORED a2a-fused kernel compile (fwd+bwd) in the maxtext image
for the v7x backend? Uses virtual tpu7x topology (no hardware). Validates vendoring + jax/pallas compat
BEFORE the moe.py surgery. EP=16 to match the 4x4x16/EP16 baseline; deepseek prod shapes."""
import numpy as np, jax, jax.numpy as jnp, time
from jax.experimental import topologies
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from maxtext.kernels.a2a_fused import fused_layer_vjp

topo = topologies.get_topology_desc("tpu7x:4x4x4", platform="tpu")  # 128 virtual v7x devices
EP = 16
devs = np.array(topo.devices[:EP]).reshape(1, EP)
mesh = Mesh(devs, ("fsdp", "expert"))
MN = mesh.axis_names
MS = dict(zip(mesh.axis_names, mesh.devices.shape))
D, F, E, K, T, BLK = 7168, 2048, 256, 8, 256, 512   # deepseek prod (E=256, k=8), small T
CAP = T * K

def core(x, idx, wts, w1, w2):
    layer = fused_layer_vjp.make_fused_moe_layer(
        ep=EP, num_experts=E, cap_rows=CAP, blk=BLK,
        ep_axis="expert", mesh_axis_names=MN, mesh_shape=MS,
        recompute_residuals=True, dw_impl="tgmm", self_last=True)
    return layer(x, idx, wts, w1, w2)

def loss(x, idx, wts, w1, w2):
    sm = shard_map(core, mesh=mesh,
        in_specs=(P("fsdp", None), P("fsdp", None), P("fsdp", None),
                  P("expert", None, None), P("expert", None, None)),
        out_specs=P("fsdp", None), check_rep=False)
    return jnp.sum(sm(x, idx, wts, w1, w2).astype(jnp.float32) ** 2)

fn = jax.grad(loss, argnums=(0, 3, 4))
x   = jax.ShapeDtypeStruct((T, D), jnp.bfloat16)
idx = jax.ShapeDtypeStruct((T, K), jnp.int32)
wts = jax.ShapeDtypeStruct((T, K), jnp.float32)
w1  = jax.ShapeDtypeStruct((E, D, F), jnp.bfloat16)
w2  = jax.ShapeDtypeStruct((E, F, D), jnp.bfloat16)
t0 = time.time()
with jax.default_device(topo.devices[0]):
    jax.jit(fn).lower(x, idx, wts, w1, w2).compile()
print(f"FUSED_KERNEL_ISO_AOT_OK compile={time.time()-t0:.1f}s (fwd+bwd, EP={EP}, self_last)", flush=True)
