# NEXT SESSION: histogram-based balanced expert assignment (user-approved, execute directly)

GOAL: capture per-layer expert histograms on the trained gate, compute optimal pi (LPT), apply
it via table-driven dispatch, A/B vs the 6.87 real-routing baseline. Everything below is
designed; build in this order.

## 1. Scan-ys recorder (the ONLY viable capture; all cheap paths measured dead)
Dead ends with receipts (EXPERIMENT_LOG 2026-08-20/22): nnx.Variable (trace levels),
io_callback (remat effects), sow in MoE (bridge mutation), sow at deepseek layer (SCANNED-layer
intermediates are DROPPED by the trainer -- only MTP survives), jax.debug.print (works at mini
scale, hard-crashes at 512 chips: callback flood).
BUILD: counts already flow to the deepseek layer via routing.bias_updates (v4 recorder,
committed). Change deepseek.post_process to RETURN the counts (new tuple element) instead of
sowing; thread through nnx_decoders `layer_fn_wrapped` as an explicit lax.scan ys (the scan at
_apply_layers_sequentially, ~line 1151); stack -> [num_layers, 256]; return through train_step
into metrics; the per-step npz dump + GCS rsync code already exists in train.py (currently fed
by the empty suffix-collect -- point it at the new metrics key). Gate: CPU mini real-jit
(command in EXPERIMENT_LOG 2026-08-20, deepseek3-671b + override_model_config tiny dims) -- it
catches everything AOT misses, minutes per cycle. Config flag: record_expert_histogram=true.
LOG-CAPTURE BUG to fix first: xpk_variant_logsave.sh's gsutil upload fails in-container (auth?)
-- verify with a 2-step run before trusting it.

## 2. Capture run
conv-style, trained gate: splitag2x image + record flag; ckpt0424 restore
(load_parameters_path=gs://mlperf-6-submission-us-central1/ckpt0424-fsdp/0/items); real c4
streaming; steps>=30 (profiler window trap: steps must exceed skip=5+2 or override profiler);
NO checkpoint_skip_step_zero_save (not in this branch). Histograms -> expert_hist/step_*.npz.

## 3. Analysis (WRITTEN: analyze_expert_hist.py, committed)
Outputs: today-vs-LPT imbalance ratio per layer; early-pi-vs-k-batch bootstrap (the GBS
question); early/late drift corr. KEY INSIGHT: grouped routing does NOT block pi -- groups are
selection-time over expert IDs; placement is the dispatch table; scattering each group's 32
experts across ranks balances rank sums without touching routing semantics.

## 4. Apply pi (table-driven, no recompile)
Dispatch maps expert->rank by integer arithmetic today (ring path: sorted_experts / group_sizes
around moe.py:1515-1530 and the ring dispatch). Replace with pi[expert] lookup, pi an s32[256]
DEVICE-ARRAY input (one compiled program serves any permutation). Weights must be permuted to
match pi ONCE at load (permute the expert axis of w0/w1/wo when restoring the ckpt -- offline
script or load-time jnp.take on the expert dim). For the A/B, a STATIC pi baked as a constant is
an acceptable v1 (recompile per pi is fine for one experiment).

## 5. A/B
Baseline 6.87 (stock, triple-confirmed). Arm: same config + pi from step 3. Attribution: step
time + the barrier-AR stall metric (all-reduce.273-class, 1.660 ms/iter today) must BOTH drop.
Expectation bracket: vbal says perfect balance = 4.65; pi captures the rank-sum share of that.

## Standing context
Best: balanced 4.251/1927; real ~6.87 clean (6.656 hoist-measured, VJP since fixed, re-gate
queued). Chunking closed: c2 optimal both regimes. compute_on pinning: -246ms real/0.11 only,
loses balanced; gate on skew. Dep-bump = regime-dependent scheduler change (-0.36 balanced /
+0.39 real) -- top diagnosis target. Split-gather parked (multi-host gate). Cluster:
bodaborg-super-tpu7x-y6k via xpk_variant_logsave.sh, IMGTAG on gcr.io/tpu-vm-gke-testing/
integrate_v2:1410-up2-*, watchers via kubectl logs job-0-0 pod.
