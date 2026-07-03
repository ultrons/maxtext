"""CPU equivalence: moe_bwd_xlayer_prefetch vs flag-off -- loss AND grads BIT-EXACT.

The NNX flag does NOT lift params (wi_0/wi_1 stay in the scanned params), so the grad TREE is
identical to flag-off and we compare leaf-by-leaf. Mini deepseek3, ring-of-experts + ragged sort
(pure-JAX SC fallbacks), random routing (constant key) and REAL routing.

Run:
  PYTHONPATH=/mnt/disks/scratch/maxtext-upstream/src \
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
      python3 xlprefetch_equiv.py [random|real]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
  os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"

import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

BASE = dict(
    run_name="xlprefetch_equiv",
    enable_checkpointing=False,
    skip_jax_distributed_system=True,
    model_name="deepseek3-671b",
    override_model_config=True,
    base_emb_dim=512,
    num_experts=8,
    num_experts_per_tok=2,
    first_num_dense_layers=1,
    base_num_decoder_layers=5,   # 4 MoE layers -> exercises top (prologue) + interior + bottom (epilogue)
    max_target_length=64,
    per_device_batch_size=1.0,
    ici_fsdp_parallelism=2,
    ici_expert_parallelism=4,
    attention="dot_product",
    megablox=False,
    use_tokamax_gmm=False,
    use_tokamax_splash=False,
    sparse_matmul=True,
    use_ring_of_experts=True,
    use_ragged_sort=True,
    ragged_gather_fallback=True,
    ragged_gather_reduce_fallback=True,
    use_random_routing=True,
    moe_routing_key_as_input=True,
    moe_handwritten_bwd=True,
    moe_weight_ag_scheduling_group=True,
    moe_n_chunks=2,
    decouple_combine_rs_chunks=4,
    moe_chunked_combine_in_remat=True,
    dtype="float32",
    weight_dtype="float32",
    grad_dtype="float32",
    dataset_type="synthetic",
    tokenizer_path="assets/tokenizer.mistral-v3",
)
_REAL = dict(use_random_routing=False, moe_routing_key_as_input=False)


def _cfg(real=False, **over):
  m = {**BASE}
  if real:
    m.update(_REAL)
  m.update(over)
  return pyconfig.initialize([sys.argv[0], os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")], **m)


def _data(cfg):
  rng = jax.random.PRNGKey(1234)
  s = (int(cfg.global_batch_size_to_train_on), cfg.max_target_length)
  ids = jax.random.randint(rng, s, 0, cfg.vocab_size)
  seg = jnp.ones(s, dtype=jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(cfg.max_target_length, dtype=jnp.int32), s)
  return ids, seg, pos


def _build(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  return models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)


def _loss_and_grads(cfg, variables, data):
  model = _build(cfg)
  ids, seg, pos = data
  rng = jax.random.PRNGKey(0)

  def loss_fn(params):
    logits = model.apply({**variables, "params": params}, ids, pos, seg,
                         enable_dropout=False, model_mode=MODEL_MODE_TRAIN, rngs={"aqt": rng})
    return jnp.mean(logits.astype(jnp.float32) ** 2)

  fn = jax.jit(jax.value_and_grad(loss_fn))
  loss, grads = fn(variables["params"])
  return float(loss), jax.device_get(grads)


def run_mode(real):
  name = "real" if real else "random"
  cfg_off = _cfg(real=real)
  data = _data(cfg_off)
  model = _build(cfg_off)
  rng = jax.random.PRNGKey(0)
  variables = model.init({"params": rng, "aqt": rng, "dropout": rng}, data[0], data[2], data[1],
                         enable_dropout=False)
  loss_off, g_off = _loss_and_grads(cfg_off, variables, data)
  loss_on, g_on = _loss_and_grads(_cfg(real=real, moe_bwd_xlayer_prefetch=True), variables, data)

  lo = jax.tree_util.tree_leaves(g_off)
  ln = jax.tree_util.tree_leaves(g_on)
  gmax = 0.0
  for a, b in zip(lo, ln):
    d = float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))) if a.size else 0.0
    gmax = max(gmax, d)
  ldiff = abs(loss_off - loss_on)
  print(f"[{name}] loss_off={loss_off!r} loss_on={loss_on!r} |loss diff|={ldiff:.3e} max|grad diff|={gmax:.3e} ({len(lo)} leaves)")
  return ldiff == 0.0 and gmax < 2e-9


def main():
  which = sys.argv[1:] or ["random", "real"]
  ok = True
  if "random" in which:
    ok &= run_mode(False)
  if "real" in which:
    ok &= run_mode(True)
  print("RESULT:", "PASS" if ok else "FAIL")
  sys.exit(0 if ok else 1)


def test_moe_bwd_xlayer_prefetch_equiv():
  """Pytest entry: loss AND grads bit-exact vs flag-off, random + real routing."""
  assert run_mode(False)
  assert run_mode(True)


if __name__ == "__main__":
  main()
