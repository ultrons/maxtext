# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU equivalence test: moe_save_block_input / moe_save_sort_indices vs flag-off.

Loss must be BIT-EXACT everywhere, through the REAL model path: the DeepSeek MoE layer's
fused custom_vjp (_handwritten_moe_layer) embedded in the decoder's nn.scan -- i.e. the
exact scan(custom_vjp(...)) structure the flags' residual threading must survive (the
"forward tracers via residuals, not closures" constvar wall). Grads: bit-exact for the
int-only save (moe_save_sort_indices); for moe_save_block_input within a ~1-ulp re-fusion
gate (see _BLOCK_INPUT_GRAD_ATOL -- the saved tensor itself is bit-identical to the
replayed value; the wobble is XLA re-fusing the changed backward graph).

Mini deepseek3 on 8 CPU devices, EP=4 x FSDP=2, ring-of-experts + ragged sort (pure-JAX
SC fallbacks), moe_n_chunks=2 + decouple_combine_rs_chunks=4 + moe_chunked_combine_in_remat
(production shape). Routing modes: random routing with a constant key (production perf
config) and REAL deepseek routing (learned gate; checks the gate-gradient path is
unchanged when routing indices are saved).

Run with:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
      python3 tests/moe_save_residuals_equiv_test.py [case ...]
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

try:
  from tests.utils.test_helpers import get_test_config_path
except ModuleNotFoundError:  # run as a plain script (e.g. in-image): resolve base.yml directly

  def get_test_config_path(relative_path: str = "base.yml"):
    from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

    return os.path.join(MAXTEXT_CONFIGS_DIR, relative_path)

BASE = dict(
    run_name="save_resid_equiv",
    enable_checkpointing=False,
    skip_jax_distributed_system=True,
    model_name="deepseek3-671b",
    override_model_config=True,
    base_emb_dim=512,
    num_experts=8,
    num_experts_per_tok=2,
    first_num_dense_layers=1,
    base_num_decoder_layers=4,
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
    ragged_gather_fallback=True,  # no SparseCore on CPU: pure-JAX ragged kernels
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

# case name -> config overrides on top of BASE. The reference for each ROUTING MODE is the
# flags-off run in that mode; every flag-on run must match its mode's reference bit-exactly.
CASES = {
    "random:save_block_input": dict(moe_save_block_input=True),
    "random:save_sort_indices": dict(moe_save_sort_indices=True),
    "random:both": dict(moe_save_block_input=True, moe_save_sort_indices=True),
    "real:save_block_input": dict(_real_routing=True, moe_save_block_input=True),
    "real:save_sort_indices": dict(_real_routing=True, moe_save_sort_indices=True),
    "real:both": dict(_real_routing=True, moe_save_block_input=True, moe_save_sort_indices=True),
}

_REAL_ROUTING_OVERRIDES = dict(use_random_routing=False, moe_routing_key_as_input=False)


def _cfg(**overrides):
  merged = {**BASE}
  if overrides.pop("_real_routing", False):
    merged.update(_REAL_ROUTING_OVERRIDES)
  merged.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **merged)


def _data(cfg):
  rng = jax.random.PRNGKey(1234)
  s = (int(cfg.global_batch_size_to_train_on), cfg.max_target_length)
  ids = jax.random.randint(rng, s, 0, cfg.vocab_size)
  seg = jnp.ones(s, dtype=jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(cfg.max_target_length, dtype=jnp.int32), s)
  return ids, seg, pos


def _build(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
  return model


def _init_vars(cfg, data):
  model = _build(cfg)
  ids, seg, pos = data
  rng = jax.random.PRNGKey(0)
  return model.init({"params": rng, "aqt": rng, "dropout": rng}, ids, pos, seg, enable_dropout=False)


def _loss_and_grads(cfg, variables, data, want_hlo=False):
  """value_and_grad of mean(logits^2) wrt ALL params, through the scanned decoder."""
  model = _build(cfg)
  ids, seg, pos = data
  rng = jax.random.PRNGKey(0)

  def loss_fn(params):
    logits = model.apply(
        {**variables, "params": params},
        ids,
        pos,
        seg,
        enable_dropout=False,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": rng},
    )
    return jnp.mean(logits.astype(jnp.float32) ** 2)

  fn = jax.jit(jax.value_and_grad(loss_fn))
  hlo = None
  if want_hlo:
    hlo = fn.lower(variables["params"]).compile().as_text()
  loss, grads = fn(variables["params"])
  return float(loss), jax.device_get(grads), hlo


# Gradient tolerance for cases that save the BLOCK INPUT (moe_save_block_input): the saved
# tensor is BIT-IDENTICAL to the backward-replayed value (verified directly: max|saved -
# replayed| == 0.0 inside fused_bwd on this harness), and the loss is bit-exact. The residual
# grad wobble (~1e-10 f32 here) comes from XLA RE-FUSING the changed backward graph (the MoE
# recompute's input is a loop-carried residual instead of a locally computed value -> different
# fusion/reduce orders for the same math) -- the same class as other fusion-affecting flags
# (cf. moe_direct_rs's documented bf16 reduce-order equivalence). Cases that only save INT
# routing tensors (moe_save_sort_indices) must be exactly bit-identical, and are.
_BLOCK_INPUT_GRAD_ATOL = 2e-9


def _compare(name, ref, out, failures, grad_atol=0.0):
  loss_r, grads_r = ref
  loss_o, grads_o = out
  loss_diff = abs(loss_r - loss_o)
  leaves_r = jax.tree_util.tree_leaves(grads_r)
  leaves_o = jax.tree_util.tree_leaves(grads_o)
  assert len(leaves_r) == len(leaves_o), f"{name}: grad tree mismatch"
  gmax = 0.0
  nonzero = 0
  for a, b in zip(leaves_r, leaves_o):
    d = float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))) if a.size else 0.0
    gmax = max(gmax, d)
    if float(np.max(np.abs(np.asarray(a, np.float64)))) > 0.0:
      nonzero += 1
  print(
      f"  {name}: |loss diff|={loss_diff:.3e}  max|grad diff|={gmax:.3e} (gate {grad_atol:.0e}) "
      f"({len(leaves_r)} leaves, {nonzero} with nonzero ref grad)"
  )
  if loss_diff != 0.0:
    failures.append(f"{name}: loss NOT bit-exact ({loss_r!r} vs {loss_o!r})")
  if gmax > grad_atol:
    failures.append(f"{name}: grads exceed gate (max|diff|={gmax:.3e} > {grad_atol:.0e})")


def _count_sorts(hlo_text):
  """Number of HLO sort ops in the compiled module (argsort/top-k lower to sort)."""
  import re

  return len(re.findall(r"\bsort\(", hlo_text))


def main():
  wanted = set(sys.argv[1:]) or set(CASES)
  failures = []
  refs = {}  # routing mode -> (loss, grads)
  hlo_sorts = {}

  data = _data(_cfg())
  variables = _init_vars(_cfg(), data)

  for mode, real in (("random", False), ("real", True)):
    if not any(c.startswith(mode + ":") for c in wanted):
      continue
    over = dict(_real_routing=True) if real else {}
    loss, grads, hlo = _loss_and_grads(_cfg(**over), variables, data, want_hlo=True)
    refs[mode] = (loss, grads)
    hlo_sorts[f"{mode}:off"] = _count_sorts(hlo)
    print(f"reference [{mode} routing, flags off]: loss={loss!r}  HLO sorts={hlo_sorts[f'{mode}:off']}")

  for case in sorted(wanted):
    over = dict(CASES[case])
    mode = case.split(":", 1)[0]
    saves_sort = "sort_indices" in case or case.endswith("both")
    saves_block = "block_input" in case or case.endswith("both")
    loss, grads, hlo = _loss_and_grads(_cfg(**over), variables, data, want_hlo=saves_sort)
    _compare(case, refs[mode], (loss, grads), failures, grad_atol=_BLOCK_INPUT_GRAD_ATOL if saves_block else 0.0)
    if hlo is not None:
      hlo_sorts[case] = _count_sorts(hlo)
      print(f"    HLO sorts: {hlo_sorts[case]} (flags-off: {hlo_sorts[f'{mode}:off']})")
      # Sort-skip receipt: only enforced in REAL routing mode. With random routing + a constant
      # key (moe_routing_key_as_input) the whole int index chain is a compile-time constant, so
      # CPU XLA constant-folds the argsorts in flag-off too and there is nothing left to skip.
      if saves_sort and mode == "real" and hlo_sorts[case] >= hlo_sorts[f"{mode}:off"]:
        failures.append(
            f"{case}: expected FEWER HLO sort ops than flag-off "
            f"({hlo_sorts[case]} vs {hlo_sorts[f'{mode}:off']}) -- sort recompute not skipped?"
        )

  print()
  if failures:
    print("FAILURES:")
    for f in failures:
      print(f"  {f}")
    raise SystemExit(1)
  print("ALL CASES PASS: loss bit-exact everywhere; sort-only grads bit-exact; "
        "block-input grads within the re-fusion noise gate")


def test_moe_save_residuals_equiv():
  """Pytest entry point."""
  main()


if __name__ == "__main__":
  main()
