# SESSION HANDOFF 2026-07-03 ~17:00 UTC (resume from here — everything below is current)

**BEST: 15.233s** = ds-v3-r8comp-0703 config: `moe_n_chunks=2 decouple_combine_rs_chunks=4
moe_routing_key_as_input=True moe_handwritten_bwd=True moe_weight_ag_scheduling_group=True
moe_direct_rs=True` on image rung8drs-64e2fa604. Loss reference: 12.270@1 -> 9.324@19 (chunk=2
lineage; chunk=1 lineage ends 9.326). Acceptance = digit-exact.

## Immediate next action (was mid-flight)
1. **Splash-offload diagnosis**: ds-v3-splashoff-0703 (image splashoff-99743694e, best stack +
   moe_splash_host_offload=True) = CORRECT (9.324 exact — LSE contract + nnx-remat bypass work on
   silicon) but **15.47 (+0.24 vs 15.23)** — the −630ms recompute win is being eaten by exposed
   transfers. Its xplane: gs://sivaibhav-exp/maxtext/ds-v3-splashoff-0703/tensorboard/plugins/profile/2026_07_03_16_48_49/
   (local pull glitched — re-pull). DIAGNOSE: (a) is splash_mha_fwd_segmented_residuals GONE from the
   bwd layer (offload engaged)? (b) are copy-start/copy-done (PCIe) spans exposed on the critical path
   (prefetch too late)? (c) compare vs r8comp-15.23s-BEST capture. Suspects: scheduler under
   xla_tpu_pcie_bandwidth_multiplier=0.03 (try raising/removing PER-RUN), no explicit prefetch
   (head scan structure). Expected win if fixed: −630ms -> ~14.6.
   **QUEUED pre-clear: ds-v3-splashoff-pcie1-0703** (same config, xla_tpu_pcie_bandwidth_multiplier=1.0)
   — sits suspended in siv-queue, auto-admits after r9e. ON RESUME read BOTH verdicts:
   r9e (dispatch compaction vs 15.23) and splashoff-pcie1 (vs 15.47/15.23 — if ~14.6-14.9 the
   scheduler's PCIe model was the blocker; if still ~15.4 the issue is missing prefetch -> profile it).
   Delete both jobsets after reading.

## Queue (order)
2. Route B A/B **LAUNCHED pre-clear**: jobset ds-v3-r9e-0703 (image rung9e-db413add5, best stack +
   decouple_dispatch_chunks=4). READ ITS VERDICT FIRST on resume: kubectl logs job/ds-v3-r9e-0703-slice-job-0
   | grep 'completed step: 19' — judge vs 15.23 (digit loss 9.324). Then delete the jobset.
3. Rung-8 AG fix: in the flat combine bwd, replace N per-chunk cotangent AGs with ONE tiled AG +
   reorder (g_out contiguous) — fixes the +0.44 remat-extension tax (receipts: profile compare in log).
4. Wag-backward port (splash_wag option B, in-kernel weight-AG in bwd recompute): the ~14ms/layer bwd
   weight re-gather stall; old branch −0.82s. After that: sort-indices/router-probs device-save (0.4GB).
5. N-sweep + 2048-slot auto chunk sizing on final stack; then record run + PR packaging
   (chunk-on-head / routing-key / manbwd+schedgroup / direct_rs are PR-shaped).

## Infra / conventions
- Cluster: jakeriley 4x8x8 via Kueue labels (queue-name: siv-queue, priority-class: maxtext-high,
  suspend: true) — canonical manifest: /mnt/disks/scratch/manifests/canonical_best_manifest.yaml (+ splashoff/r8comp/e8adlo variants there); runbook
  ~/maxtext/CLUSTER_RUNBOOK.md (sharing w/ micro agent, gotchas: scoped-vmem 65472 when SC wrappers
  present, trace flag unusable, digit-exact loss acceptance).
- Secondary rig: sivaibhav-exp-v7x (2x2x2 spots + 2x2x1 rigs + 4x8x8/4x4x4 flex; hl-* tenants).
  bodaborg-sps ABANDONED.
- Branches (worktree /mnt/disks/scratch/maxtext-upstream): rung6-sliced = mainline (HEAD ~db413add5+);
  splash-offload branch @ 99743694e in /mnt/disks/scratch/maxtext-splash (MERGE into rung6-sliced when
  both settle). Old branch (~/maxtext, decouple-combine-rs-chunks) = read-only reference.
- Profiles: :9010 dropdown /mnt/disks/scratch/xprof-runs/ (r8comp-15.23s-BEST = the reference).
- USAGE DISCIPLINE (user directive): fresh sessions per phase; haiku/sonnet agents for mechanical work,
  short briefs; never resume agents >300k tokens; chain background scripts so one notification carries
  multiple verdicts; -m1/head on all tool output; terse turns.


## DO NOT REDO (settled verdicts + bug classes — receipts in sections below / memory files)
- CLOSED, never revisit without new evidence: fused SC combine+RS kernel (22.9x slower, bench-killed);
  EP=8 (fits only w/ decoder_layer_input=offload @16.77; levers INVERT there; EP4 wins >=1.5s);
  fence/scheduling levers for SC-queue serialization (shared-expert fence no-op — only ~1.4ms movable
  TC work; RS-offload-off = sync TC RS = worse; async-RS fusion flags = v7x hard-fatal);
  moe_n_chunks>2 (16.17s, GMM tax); A2A dispatch at EP4/topk8 (degenerates to AG);
  converting gmm/QKV/token-AG recomputes to save/restore (restore 3-8x worse — measured).
- BUG CLASSES (bitten 3+ times — check FIRST on any new kernel-adjacent failure):
  (1) chunk-shaped operands into kernels: never slice kernel operands OR verify every internal
  count/stride/mode derives from the RIGHT operand (stride-from-indices fix, full_num_slots pinning,
  rank-space vs buffer-position bounds — searchsorted fix 26a250f4e);
  (2) CPU tests CANNOT catch SC-kernel contract bugs (fallback ignores bounds) — numpy addressing
  EMULATION required pre-hardware, harness must mirror the model composition scan(custom_vjp(core_map));
  (3) JAX Refs cannot cross jit/custom_vjp/scan boundaries; pl.kernel has NO input_output_aliases
  (in-place SC accumulate = framework wall; validated kernel parked as reference);
  (4) XLA fuses across chunk iterations (multi-output convert fusion) — barrier-chain any pipelined
  loop; note xla_tpu_aggressive_opt_barrier_removal=true in flags can strip naive barriers.
- HARD RULES: moe_handwritten_bwd REQUIRES moe_weight_ag_scheduling_group (else 1.8-2.8T temps);
  ragged_buffer_factor stays -1 (user constraint, rbf<4 fatal under real routing);
  moe_routing_key_as_input on BOTH sides of any bwd A/B; loss acceptance is DIGIT-EXACT;
  read AOT logs BEFORE pushing images; new SC-side machinery may need scoped_vmem 65472.
- KNOWN-INERT dials: opt-offload for EP8 temps; chunking for EP8 temps; cost_estimate flags for the
  ragged kernels (already auto-computed).

## Key numbers memorized nowhere else
fwd 69.6ms/layer, bwd 180.2ms/layer (2.59x); bwd = recompute 35ms + true-grad 95ms + exposed 50ms;
splash recompute 10.9ms/layer; token-AG exposed ~720ms/step; bwd weight re-gather ~14ms/layer;
O+LSE = 545MB/layer/device = 18.2ms via 64GB/s/chip PCIe (2 cores share), store hides in 69.6ms fwd
window, restore needs 1-layer prefetch in 180ms bwd window; host-DMA lane was 95% idle pre-offload.

# MORNING SUMMARY 2026-07-03 (final, overnight autonomous run complete)

**BEST: 15.233s** = moe_n_chunks=2 + decouple_combine_rs_chunks=4 + barrier chain + moe_direct_rs
(image rung8drs-64e2fa604, run ds-v3-r8comp-0703, loss 9.324 digit-exact). Ladder: 15.74 -> 15.23
(-0.51s). Profile "r8comp-15.23s-BEST" on :9010. All levers flag-gated default-off on rung6-sliced.

Overnight arc (receipts for everything in sections below):
- Combine side DONE: chunk mechanism + barrier chain + TC-lane direct-RS (old-branch port) = the win;
  fence/flag levers exhausted with receipts; fused-SC kernel KILLED by Step-0 bench (22.9x slower).
- Rung 7+8 (remat/bwd extension): CORRECTNESS COMPLETE on silicon (flat single-vjp bwd, -16.3GB temps,
  22.5s naive tax eliminated) but perf-negative: 15.67 vs 15.23 remat-off -> flags stay OFF pending
  profile-guided iteration (daylight task).
- Rung 9 (input side): correctness PROVEN on hardware first try; perf blocked on a NAMED Pallas
  framework wall (no in-place SC accumulate: new_ref(input) = Ref-as-output forbidden; pl.kernel has
  no input_output_aliases). Kernel itself v5p-validated (diff=0, 0 copies). THREE ROUTES FOR YOUR CALL:
  (A) in-image JAX patch plumbing aliasing through pl.kernel (+ upstream feature request),
  (B) compaction-first redesign (functional, no refs; ~+3.7ms/layer cost vs ~6ms prize — needs bench),
  (C) park input side.
- 5 hardware bugs fixed w/ emulation receipts; 2 framework gotchas named (scoped-vmem 64KiB reserve
  pattern; custom_call_region_trace unusable); bodaborg reservation broken since 06-25 (bench armed).

Suggested morning order: (1) pick input-side route A/B/C; (2) profile r8final vs r8comp to see where
the remat extension pays/costs; (3) N-sweep + 2048-slot auto-sizing on the 15.23 stack; (4) upstream
PR planning for the clean rungs (chunk PR + routing-key + manbwd + direct_rs are PR-shaped).

# DS-v3 v7x Experiment Log

# MORNING SUMMARY 2026-07-03 (overnight autonomous run)

**NEW OVERALL BEST: 15.233s** (`moe_n_chunks=2 + decouple_combine_rs_chunks=4 + barrier chain +
moe_direct_rs`, image rung8drs-64e2fa604, loss 9.324 digit-exact) — ladder total 15.74 -> 15.23
(-0.51s), every rung loss-verified. Profile: xprof :9010 "r8comp-15.23s-BEST".

Scoreboard: baseline 15.74 | +c2 15.52 | +routing-key 15.37 | +manbwd 15.33 | decouple4+drs 15.46 |
**composition 15.23**. (Old records for reference: 15.49 autodiff / 14.54 manbwd+wag, old branch.)

Overnight verdicts: (1) fused SC kernel KILLED by Step-0 bench — 22.9x slower, receipts in
FCR_STEP0_BENCH_LOG.txt; (2) moe_direct_rs (old branch, TC-lane RS) ported+validated = the win;
(3) two hardware bugs fixed w/ emulation receipts (fwd stride, bwd rank-bounds — r7 chunked-remat now
CORRECT but naive-bwd expensive 22.5s, rung-8 single-vjp bwd port is the fix); (4) fence/flag levers
on SC serialization exhausted with receipts; (5) chunk-2-on-head is genuinely good (free cross-engine
RS hiding — old branch premise doesn't replicate).

NEXT (recommended order): (a) INPUT side rungs 9-11 — token-AG ~720ms/step exposed in EVERY config,
design agreed (full-shape windows + disjoint in-place writes + custom_vjp fence) + SC-block-slide
second payoff; (b) rung-8 memory-flat bwd port (makes chunked-remat usable, extends wins to bwd);
(c) N-sweep + auto chunk sizing on the composition config. Open infra: bodaborg reservation broken
since 06-25 (bench job left armed there).


Working cluster: `jakeriley-v7x-do-not-delete` (project `tpu-prod-env-automated`, region `us-central1`,
`--dns-endpoint`; context `gke_tpu-prod-env-automated_us-central1_jakeriley-v7x-do-not-delete`).
Hardware: dedicated nodepool `siv-tpu-4x8x8` — 64x `tpu7x-standard-4t` = one 4x8x8 slice (256 chips,
512 cores), reservation `cloudtpu-20260319010000-1695218634`. Kueue is bypassed (its only flavor is
2x2x1-scoped): jobsets applied WITHOUT `kueue.x-k8s.io/queue-name`, manifests hand-patched from xpk
dry-runs (xpk lacks tpu7x-4x8x8: dry-run at 4x4x8 → patch parallelism/completions 32→64, topology
label → 4x8x8, DROP the stale `placement-policy-name` selector).

Bucket: `gs://sivaibhav-exp/maxtext/<run>/` (logs under `logs/`, xplane under `tensorboard/<run>/plugins/profile/<ts>/`).
Manifests: session scratchpad `bl256_manifest.yaml` etc. Launchers: `~/maxtext/baseline_256_launch.sh`.
Local full log copies: `/mnt/disks/scratch/<run>.log`.

Base config (all runs unless noted): deepseek3-671b 61L, PBS4, seq4096, EP4 x FSDP128, rbf=-1,
chunk=1, synthetic data, random routing, adamw bf16, remat=custom, tokamax gmm_v2 + splash,
flags_A.txt XLA flags, steps=20, profiler=xplane steps 6-8.

## Images

| tag (gcr.io/tpu-prod-env-automated/integrate_v2:) | contents | libtpu |
|---|---|---|
| `baseline-upstream-fcb7ebeb` | UPSTREAM AI-Hypercomputer/maxtext main @ fcb7ebeba (2026-07-02) | 0622-nightly .so (from xlayer-hierrs-nl base — the 15.49 record libtpu) |
| `dcrs-c2e46b7f-nl` | OUR branch decouple-combine-rs-chunks @ c2e46b7f1 (manbwd + levers, default-off) | same 0622-nightly base |

## Runs

### ds-v3-bl256-c1oo-0702 — upstream baseline, chunk=1 + opt-offload  [2026-07-02]
- image `baseline-upstream-fcb7ebeb`; delta flags: `optimizer_memory_host_offload=True`
- **step time 16.13 s** steady state (steps 11–19: 16.129–16.132s, 3ms spread; steps 5/8 ~32s =
  profiler start/stop, step 6 0.008s = async metric artifact, step 10 186s = profile upload stall)
- ~254 TFLOP/s/device; loss 12.27→10.55 by step 8, smooth, no NaN; moe_lb_loss=0
- vs 15.49s record (same topology/libtpu, no offload): **+0.65s (+4.2%)** — attribution pending
  (offload flag vs upstream code drift; see bl256-c1 + ref256)
- profile: steps 6–8 → `gs://sivaibhav-exp/maxtext/ds-v3-bl256-c1oo-0702/tensorboard/` (to xprof :9010)
- GCS write-probe from worker pod: OK (`wprobe.txt` — cleaned up after profile pull)

### ds-v3-bl256-c1-0702 — upstream baseline, NO offload  [2026-07-02, COMPLETE]
- identical to bl256-c1oo minus `optimizer_memory_host_offload` → exact 15.49-record config on upstream head
- **step time 15.74 s** steady (steps 11–19: 15.740–15.744s); loss identical trajectory to c1oo
- ATTRIBUTION: offload tax = 16.13−15.74 = **+0.39s (+2.5%)**; upstream drift vs 15.49 record =
  **+0.25s (+1.6%)**
- profile: steps 6–8 → `gs://sivaibhav-exp/maxtext/ds-v3-bl256-c1-0702/tensorboard/`

### manbwd 1.80T blowup investigation [2026-07-02, OPEN]
- manbwd at EP4/FSDP128 61L → **1.80T HLO temps** (AOT tpu7x-512). IDENTICAL 1.80T for: +splash_host_offload
  (mb2), old June image moe-splash-merged (mbold), no opt-offload (mbnooo) → NOT branch drift, NOT the
  offload flags, NOT splash residuals. ~29.5GB/layer = full per-layer forward saved as residual.
- mbold8 (old image, EP8/FSDP64) = **2.76T** → NOT mesh-fixable; EP8 is worse.
- **mbrec (FULL 14.54-record flag set: chunk2 + chunk_barrier + wag fwd/bwd + scheduling groups +
  sa_block_kv_dkv_compute=1024) = PASS.** mbc2 (chunk2+manbwd only) = 1.80T → chunking does NOT fix it.
- **BISECTED: the SCHEDULING-GROUP flags are the fix.** mbwag (+wag only) = 1.80T; mbgrp
  (+`moe_weight_ag_scheduling_group`+`moe_handwritten_splash_group` only) = **PASS**. → The blowup is an
  XLA SCHEDULER pathology, not custom_vjp residuals: without the annotation the scheduler co-schedules
  all 61 layers' bwd re-gathers/recomputes → 61 co-live working sets ≈ 1.8T (EP8's bigger per-shard
  buffers → 2.76T). RULE: **never run moe_handwritten_bwd without moe_weight_ag_scheduling_group**
  (single-flag confirm gate mbwagonly running).
- Chunk-independent, mesh-scaled (EP8 2.76T vs EP4 1.80T), branch-independent, offload-independent.
  Signature ~29.5GB/layer = full per-layer forward saved as custom_vjp residual.
- CONSEQUENCE: bit-exact pair runs at the record flag set (chunk=2 basis), NOT chunk=1:
  ref = autodiff + shared set; mb = + manbwd + wag + groups. Launched on cluster (see below).

### ds-v3-ref256-c2-0702 / ds-v3-mb256-c2-0702 — bit-exact pair  [2026-07-02, running]
- image `dcrs-c2e46b7f-nl`; shared: `moe_n_chunks=2 moe_chunk_barrier=True moe_routing_key_as_input=True
  sa_block_kv_dkv_compute=1024`; mb adds: `moe_handwritten_bwd moe_splash_wag_forward
  moe_splash_wag_backward moe_weight_ag_scheduling_group moe_handwritten_splash_group`
- criterion: per-step loss identical (both deterministic, same constant routing key)
- **RESULT [INVALIDATED as correctness evidence + NOT bit-exact anyway]:**
  - ref = **15.46s** steady, mb = **14.85s** steady (manbwd+wag −0.61s vs same-branch autodiff — the
    one durable datapoint; record-lineage win reproduced on new libtpu).
  - **Loss FLAT on our branch** (12.270→12.269 over 20 steps) vs upstream's healthy descent
    (12.27→10.55@8) — branch pathology, lr/opt defaults identical to upstream → suspect branch code or
    extra flags. Matching a pathological curve proves nothing (user call) → pair demoted.
  - Per-step perplexity DIFFERS ref-vs-mb from **step 0** (213159.9 vs 213157.9, ~1e-5 rel) → the
    FORWARDS differ (splash_wag_forward fp reorder or manbwd fwd-capture), not a backward-math signal.
    3-decimal loss matches; the old "bit-exact" claim may have been at that coarser grain.

## METHODOLOGY PIVOT [2026-07-02]: one change at a time on maxtext HEAD
Our branch = side reference only. Ladder, each rung one change vs previous, loss must stay on the
healthy descending curve:
- Rung 0 = bl256-c1 (upstream head, 15.74s) ✓
Full 11-rung plan (user, 2026-07-02): 1)✓ baseline 2) chunk=2 3) routing-key-as-data 4) manbwd chunk=2
5) manbwd chunk=1 6) combine-side (unsort+RS) chunking fwd-only (incl. c>0 NaN fix) 7) +bwd-remat
8) +full bwd 9) dispatch (AG+sort) pipeline fwd 10) +remat 11) +full bwd. Target ~13.65s, no rbf,
unchunked GMMs, chunk 4–8 both pipelines. Profiles → xprof :9010 compare dropdown
(/mnt/disks/scratch/xprof-runs/<run>-<time>/).

### Rung 2: ds-v3-c2head-0702 — head + moe_n_chunks=2  [2026-07-02, PASS w/ caveat]
- image `chunk2-on-head-73eb7170` (3 chunk-PR commits cherry-picked clean onto fcb7ebeba)
- **15.52s** steady (−0.22s vs rung-0 15.74 — chunking is FASTER on head)
- Loss curve HEALTHY, tracks rung 0 (12.27→9.32@19 vs 9.325). NOT strictly bit-identical: step-0
  perplexity differs ~9e-6 rel (XLA fusion/reduce-order for chunked shapes, bf16 compile noise);
  drift ≤0.003 loss by step 19. Standard adopted: trajectory-identical within compile noise.
- BONUS bisect datum: chunk code does NOT reproduce our branch's flat-loss pathology.

### Rung 3: ds-v3-r3key-0702 — + moe_routing_key_as_input  [launched]
- image `rung3-aa8aa5526`: minimal hand-written port (22 lines: base.yml+types.py flags + get_topk
  constant-seed branch). NOT the branch's threading design — key derived in-scope at get_topk
  (recomputable by rung-4 manbwd by construction). Cherry-pick of branch commits REJECTED (4c3ddf98a
  bundles manbwd scaffolding — violates one-change).
- NOTE: constant key ⇒ routing FROZEN across steps (branch behaved the same). Acceptance:
  convergence statistically similar to rung 2, NOT bit-exact (routing pattern changes).
- **RESULT: PASS — 15.37s steady; loss curve IDENTICAL to rung 2 at every printed digit (12.270→9.323).**
  Frozen-key routing indistinguishable from rng routing on this workload. Profile → r3key-15.37s.
  (Also: our branch's flat loss NOT caused by routing-key mechanism either — bisect narrows further.)

### Rung 4: ds-v3-r4mb-0702 — + manbwd (chunk=2)  [2026-07-02, PASS]
- image `rung4-f622dd2b2` (branch rung4-manbwd @ f622dd2b2, agent port): hoist-path fused_bwd only,
  flags moe_handwritten_bwd + moe_weight_ag_scheduling_group; annotated gather custom_vjp (fwd AG /
  bwd psum_scatter, sched groups 1/2/3); decoders.py set_remat_policy SKIP under manbwd (agent-found,
  essential — auto-remat cycles against the annotation otherwise); NO prefetch machinery (2-arg fused);
  NO wag/stagger/host-offload variants (later rungs). Flag-off AOT byte-safety PASS.
- **15.33s steady** (−0.04 vs rung 3); loss within compile noise of rung 3 (step19 9.324 vs 9.323,
  drift ≤0.002 — same scale as rung-2 fusion noise; wrong gradients would diverge, not track).
- Compiled clean on cluster — NO 1.8T (scheduling-group annotation port works on head).
- Agent-flagged review items: (1) gather bails on prefuse_moe_weights entirely; (2) training-only
  (primal ignores previous_chunk/slot — do NOT enable for chunked-prefill inference); (3) manbwd gate
  requires rng-free routing = use_random_routing→moe_routing_key_as_input (get_topk constant key).
- Profile → r4mb-15.33s.

### Rung 5: ds-v3-r5mbc1-0702 — manbwd chunk=1  [launched]
- same image, flag flip moe_n_chunks 2→1. Tests whether manbwd needs chunking on head (the old-branch
  chunk=1 manbwd was never memory-viable; with the annotation it may be now — cluster compile decides).

### ds-v3-ref256 / ds-v3-mb256 — bit-exact manbwd pair  [gating]
- image `dcrs-c2e46b7f-nl`; both add `moe_routing_key_as_input=True` (constant routing key — REQUIRED
  on BOTH sides; upstream image derives routing differently → never bit-exact cross-image)
- ref = autodiff; mb = + `moe_handwritten_bwd=True`
- criterion: per-step loss identical ref vs mb
- AOT gates (tpu7x-512, local): running

## AOT gate results (local train_compile, compile limit 94.74G)

| gate | shape | config | verdict |
|---|---|---|---|
| 128-chip EP4xFSDP64 PBS4 chunk1 | tpu7x-256 | — | OOM by 96MB |
| 128-chip + opt-offload | tpu7x-256 | upstream + ours both | PASS (~0.7G headroom) |
| 128-chip chunk=2 | tpu7x-256 | ours | PASS (~1.9G headroom) |
| 64-chip EP4xFSDP32 PBS4 (all lever combos) | tpu7x-128 | g1/g2/g3 | ALL FAIL (temps 121–127G) — 64 chips needs PBS2 |
| 256-chip ref/mb pair | tpu7x-512 | ours | pending |

## Dead ends / gotchas (this campaign)
- nap cluster (`bodaborg-tpu7x-auto-nap2`): 4x4x8 NAP scale-up failed twice (`Internal error` ~ 4t
  stockout us-central1-c); only 4x4x4 slices physically exist there. 128-chip jobs deleted.
- Priority: xpk caps at very-high(1000); 2000 tier = k8s priorityclass `poc-ml-perf-priority`
  (patch manifest). Preemption LowerPriority works (evicted a 1000-tier ubench in <1 min).
- `gcr.io/...:decouple-combine-rs-fix` tag was PRUNED; use `-fix2`.

### Rung 5: ds-v3-r5mbc1-0702 — manbwd chunk=1  [2026-07-02, PASS — constraint retired]
- same image as rung 4, moe_n_chunks=1. **COMPILES AND RUNS** — the old-branch "manbwd needs chunk=2"
  memory constraint was ALWAYS the missing scheduling-group annotation, never chunking.
- **15.61s** steady (slower than chunk=2's 15.33 — chunk=2 stays the perf choice); loss 9.326@19, in-noise.
- Profile → r5mbc1-15.61s.

### Rung 6: ds-v3-r6dcrs4-0702 — combine-side (unsort+RS) chunking N=4, fwd-only  [launched]
- image `rung6-b71601ee1` (branch rung6-combine-chunks @ b71601ee1, agent port+fix): flag
  `decouple_combine_rs_chunks=4`, moe_n_chunks=1 (NO chunking elsewhere, per plan), manbwd recompute
  uses UNCHUNKED combine (fwd-only rung; rung 7 flips the remat).
- **ROOT-CAUSE CORRECTION (the old "c>0 NaN"):** NOT `_permute_tokens_for_chunked_rs` (verified
  bit-exact) — the bug was `ring_ragged_unsort`'s packed-vs-global buffering-mode decision using the
  chunk SLICE length (`num_tokens = revert_indices.shape[0]`) instead of the FULL problem's slot count:
  per-chunk calls flipped into global-positions mode over a PACKED buffer → misaligned shard combines
  (~100-530 magnitudes from ±0.001 data), indices in-range. Fix: thread `full_num_slots` (optional arg,
  default = old behavior → unchunked byte-identical). Truncated+chunked under autodiff bwd now raises
  NotImplementedError loudly. The old EP=4 "bit-exact" test missed it because the mode only flips when
  buffer_size >= slots_per_chunk (rbf/EP/N-dependent).
- Equivalence test (tests/chunked_combine_equiv_test.py, CPU x8): max|diff|=0.0 across EP{4,8} x
  {non-uniform,uniform} x N{1,2,4,8} x {full,truncated}; autodiff grads 0.0; real RoutedMoE v4 EP=2
  bit-exact; CPU real-jit tiny train step flag-on == flag-off (12.290), no tracer escape.
- Acceptance: loss in-noise vs rung 5 (15.61s, 9.326@19) — same math, new schedule; step time = does
  the RS hide under per-chunk combine (profile check vs r5 if unclear).

### Filler: ds-v3-r4mbc4-0702 — manbwd moe_n_chunks=4  [2026-07-02]
- **16.17s** (loss 9.324, in-noise). moe_n_chunks curve: c1=15.61 / c2=15.33 / c4=16.17 —
  full-pipeline chunking beyond 2 is NEGATIVE (GMM MXU-efficiency loss > overlap gain).
  VALIDATES the ladder thesis: chunk-4/8 pipelining must be DECOUPLED from the GMMs (rungs 6-11);
  the 13.65 target is unreachable via moe_n_chunks.

### Rung 6 v2 [in progress]: full-shape + valid-window redesign (user design)
- All three rung-6 failures traced to SLICED ridx/w operands (chunk-shaped kernel calls): the fixed
  mode-flip bug, the N-independent SC NaN, the fallback Mosaic tile misalignment. Source impl was
  half-windowed: buffer full-shape, indices sliced.
- v2 design: operands FULL-shape every invocation; chunk = contiguous valid window in final token
  order, expressed via the kernels' EXISTING start/end row bounds (as the bwd already does with
  shard_output_start/end); per-chunk OUTPUT slice -> psum_scatter. Mutable-array note: jax 0.10 refs
  available for windowed output writes (nicety, not the fix).

### Rung 6 v2 verdict: ds-v3-r6v2dcrs4-0702  [2026-07-03, CORRECTNESS PASS / PERF FAIL]
- image rung6-7fd97ea51 (window formulation + kernel stride fix + fallback dtype fix)
- **NaN SAGA CLOSED**: no NaN, loss 9.326@19 digit-identical to rung 5. Root causes (agent, receipts):
  (1) SC NaN = kernel `row_partition_size` derived from BUFFER rows instead of INDICES rows (silent OOB
  reads of slot arrays, bounds checks off; N-independent) — one-line fix; NB tpu-inference's copy of
  this kernel always had it right (indices_hbm_ref.shape[0]). (2) fallback Mosaic error = PRE-EXISTING
  bf16*f32 dtype promotion in the fallback gather (f32 cotangent -> gmm-bwd tiling violation) — also
  chunking-independent (control-run proven on old image).
- **PERF: 16.95s vs rung-5 15.61 (+1.34s). Profile compare: SC LANE +1.32s** = N x full-slot-array
  preprocessing (window formulation invokes full-shape kernel N times; per-call fixed cost O(T)).
  TC compute unchanged; VPU +180ms.
- A/B in flight: rung6-sliced variant (tpu-inference-style sliced operands on the FIXED kernels —
  per-call preprocessing O(T/N)). Trade: v2 elegance/shape-invariance vs v1 per-call cost. Note v2's
  "no kernel change" premise didn't hold anyway (stride fix was required regardless).
- tpu-inference comparison (user q): same pipelining lineage (identical _permute_tokens_for_chunked_rs),
  sliced operands + caller-computed mask operand + empirical 2048-slot chunks + optional hier-RS kernel;
  inference-only (no bwd). Steal later: 2048-slot auto chunk sizing, hier-RS for the scatter leg.

### Rung 6 forensics verdict [2026-07-03] — why the RS doesn't hide (receipts in agent report)
- The 4 chunk combines run GAPLESS on SC (4x2.3ms = 9.2ms == unchunked 9.3ms; chunking itself is free).
- BLOCKER 1 (fusion): XLA merged the 4 per-chunk f32->bf16 converts into ONE multi-output
  convert_select_fusion consuming all 4 combine outputs -> every chunk's RS depends on the LAST
  combine; all 4 RS starts batch after combine #4. Fix = per-chunk barrier/structural convert
  (rung6b image in progress).
- BLOCKER 2 (engine): SC-offloaded RS shares the SC queue with the combines (can't overlap own cover);
  offload-OFF lowers RS as SYNC on TC (blocks TC). Need async ICI RS (flag hunt) or the SC-fused
  combine+RS kernel (structural, ranked most robust).
- Exposed token-RS ~740ms/step; token-AG sibling ~720ms/step (rungs 9-10 prize).
- Chunked RS pays +18% (smaller messages): 7.6ms vs 6.4ms per layer at N=4.
- NOTE: --xla_enable_custom_call_region_trace=true is UNUSABLE on this config: needs scoped_vmem
  <=65472KiB AND then core-halts (scheckne) at runtime — instrument perturbs execution. Overlap
  forensics done from uninstrumented captures instead.

### Rung 6 barrier matrix [2026-07-03 overnight]
- r6b1 (barrier un-fuse, SC-offload ON): **15.61s**, loss 9.326 — barrier recovered the +30ms overhead
  (exact parity with unchunked). RS still exposed (shares SC queue with combines).
- r6b2 (offload OFF + async-RS-fusion flags): **REJECTED by libtpu** — hard fatal: "Continuation fusion
  for ReduceScatter... not supported on platforms other than Viperlite... incorrect results, hang, SDC".
  **Async-ICI-RS via flags is PLATFORM-DEAD on v7x.**
- PIVOT: shared-expert-as-RS-cover (structural): keep RS async on SC, deadline-barrier the shared-expert
  GMM (TC, ~5-6ms — same size as the 6.4ms RS) into the RS window. Flag moe_shared_after_combine,
  image rung6c-* (agent implementing). Fused-SC combine+RS kernel remains the reserve.
- Other ideas logged for later: hier-RS TC-kernel as chunk RS (cross-engine without flags);
  causal-chunk cross-layer pipelining (blue sky); A2A dispatch REJECTED at EP4/topk8 (degenerates to AG).

### Shared-expert cover matrix [2026-07-03 overnight, NULL]
- r6c1 (N=4 + moe_shared_after_combine): **15.609s** — zero change vs r6b1. r6c2 (N=2): 15.658s.
  Loss 9.326 both (fence is correctness-clean).
- Two hypotheses (analyst dispatched on r6c1 capture): fence stripped by aggressive_opt_barrier_removal,
  OR fence held but is only a lower bound (scheduler still parks shared GMM after the RSs), OR the
  deeper possibility: with combines+RSs BOTH on the SC queue, the SC lane (~17ms serial) is the layer's
  binding lane — in which case NO TC scheduling helps and the only fixes are (a) RS on another engine
  (hier-RS TC kernel) or (b) RS hidden INSIDE the SC kernel (fused combine+RS, scalar-subcore DMA ||
  vector-subcore compute — SC lane collapses to ~max(9.2, RS) instead of 9.2+7.6).
- Awaiting analyst verdict on (d): SC-lane vs TC-lane per-layer totals — decides hier-RS vs fused-SC.

### Forensics round 2 [2026-07-03 overnight] — fence levers EXHAUSTED, receipts in
- Barrier chain CONFIRMED working (fusion split, staggered RS starts, chunk-0 RS stall 1.89->0.49ms
  = ~10ms/step). Shared-expert fence = behavioral no-op: XLA already had gate/up in the RS window,
  and the movable TC work is only ~1.4ms (my 5-6ms premise was 4x off — measured).
- **Layer tail = 17.3ms SAME-ENGINE SC serial chain** (9.5 combines + 7.8 RS drain), gate = RS3-done
  -> residual add. TC idles ~14ms there but filling it CANNOT shorten the layer. No schedule fixes
  this without moving/shrinking/fusing the RS.
- Ranked (analyst): (1) fused SC combine+RS kernel: SC tail -> ~10ms, ~-350ms/step fwd [SCOPING AGENT
  DISPATCHED -> FUSED_KERNEL_PORT_PLAN.md]; (2) slide SC block under gmm-down via INPUT-side chunking
  (= rungs 9-11, extra payoff up to -5.6ms/layer); (3) bf16 combine output (halve SC write + kill
  converts, modest).
- Ladder continues meanwhile: rung 7 (remat flip) in progress on rung6-sliced.

### Fused SC combine+RS — scoping verdict [2026-07-03, plan: /mnt/disks/scratch/maxtext-upstream/FUSED_KERNEL_PORT_PLAN.md]
- Port tractable: old branch has complete A3 kernel (402 LOC) + wiring; A3-bwd NOT needed under manbwd;
  SC-offload bypass automatic (no psum_scatter HLO emitted); found+documented an EP>1 clamp
  double-count bug in the old adapter (needs zero-row pad fix).
- **BLUNT RISK: the perf thesis is unproven at production shape** — kernel combine uses 1 vector
  subcore vs mpmd's 32, and f32-only RS moves 2x wire bytes (worst case 15.6ms DMA under 9.5ms compute
  = un-hideable). Old "pass" was NTOK=64 = 256x under production (documented trap).
- STEP 0 dispatched: thesis-kill bench on bodaborg-tpu7x-sps (TEC-only vs mpmd throughput; f32 RS
  hiding ratio; projected SC-tail vs measured 17.3ms). Kills or confirms before any port code.

### Rung 7 [2026-07-03 overnight, FAILED — bwd hardware bug, isolated]
- r7 (chunked_remat + shared_after): NaN @ step 1. r7b (chunked_remat only): NaN @ step 1, step-0
  loss exact 12.270 -> fence exonerated; the chunked-combine BACKWARD is broken on hardware (CPU
  digit-exact — hardware-only, the fwd stride bug's mirror: chunk-sized cotangent vs full-sized
  buffer/index operands into SC ragged_gather with bounds checks off). First hardware contact of this
  bwd path (rung 6 fwd-only never ran it). Dev agent auditing kernel-contract derivations + building
  a bwd addressing emulation; flag stays default-off; rungs 1-6 stack unaffected.

### Fused-kernel Step-0 bench [2026-07-03 overnight, INFRA-BLOCKED -> fallback in motion]
- bodaborg-tpu7x-sps: ZERO TPU nodes — ghostfish reservation fault (CONDITION_NOT_MET, failing since
  06-25, all 16 pools stuck, no autoscaler path). Bench job `fcr-step0-bench` left ARMED there
  (runs unattended if reservation heals; kubectl logs job/fcr-step0-bench, grep TIMING|DERIVED).
- Fallback dispatched: same ConfigMap+Job -> sivaibhav-exp-v7x (tpu-vm-gke-testing; had a live
  single-host v7x on 06-29/30 per tstat_results.md receipts).
- Bench design note: fused kernel gathers ALL slots (no validity compaction) -> honest comparison is
  fused-all-valid vs ragged_gather_reduce-quarter-valid; also measures psum_scatter f32/bf16 refs +
  correctness vs jnp reference. Thesis verdict: NEEDS RE-MEASURE (zero numbers exist).

### Step-0 bench VERDICT [2026-07-03, ran on sivaibhav-exp-v7x]: fused SC kernel DEAD as-is
- Measured (receipts: /mnt/disks/scratch/maxtext-upstream/FCR_STEP0_BENCH_LOG.txt): fused TEC-only
  combine 117.4ms vs mpmd ragged_gather_reduce 5.14ms at identical inputs = **22.9x slower**
  (1 subcore vs 32, flat 16 GB/s DMA-issue-bound; shape-independent). RS exposure 5.6-11.1ms alone
  > TC psum_scatter_f32 (1.97ms). Projected production tail ~220ms vs the 17.3ms it must beat.
  BONUS: latent correctness bug at CH=4096 (rel=0.93; exact at CH=2048) — never validated there.
- Port CANCELLED. Required 23x IS the 32-subcore partitioning rgr already has -> viable architecture
  would be SCS-RS-streamer grafted onto rgr = kernel-track R&D, out of ladder scope.
- **NEW LEVER FOUND: old branch's `moe_direct_rs`** — validated TC-Pallas direct-to-owner RS w/
  custom_vjp (==psum_scatter). TC-engine RS = no SC-queue conflict + immune to v7x async-RS
  prohibition + fits the measured ~14ms idle-TC window. Projected tail 17.3->~11.5ms ≈ -0.34s/step
  fwd. Port dispatched (rung8drs-*).

### Rung 7 fix verdict: r7c [2026-07-03] — CORRECTNESS PASS / perf as-expected-bad
- rank-space bounds fix (26a250f4e) works on hardware: loss digit-identical (12.270->9.326).
- **22.5s/step** = the naive per-chunk bwd (~N full-buffer SC ragged_gathers vs 1). EXACTLY the
  rung-8 motivation: port the old branch's single-custom_vjp unchunked-bwd-applied-once design
  (chunked_ring_combine_reduce_scatter bwd: per-chunk AG of g_out -> concat -> ONE ragged_gather).
  Rung 7 flag stays off in any perf config until rung 8 lands.

### moe_direct_rs A/B [2026-07-03] — FIRST COMBINE-SIDE WIN
- r8drs (chunked N=4 + barriers + TC-lane direct-RS): **15.46s**, loss 9.326 digit-exact. −0.15s vs
  15.61 control. All port distrust items cleared (no collective_id hang, addressing correct).
  About half the −0.34s projection: last chunk's RS trails + production ICI ≠ single-host TC-RS rate.
- Matrix completing: r8drsu (UNCHUNKED + direct_rs — kernel effect alone) and r8drsc2
  (moe_n_chunks=2 rung-4 config + direct_rs — does it push the 15.33 overall best down).

### direct-RS matrix complete [2026-07-03] — honest scoreboard
- unchunked+drs 15.79 (REGRESSION: no overlap target, TC-RS blocks TC) | decouple4+barriers+drs
  **15.46** (the win) | moe_n_chunks=2+drs 15.50 (REGRESSION vs 15.33: chunk-2's SC-offloaded RS
  already hides under TC GMM compute cross-engine FOR FREE; TC-RS steals from GMMs).
- **OVERALL BEST remains rung-4 moe_n_chunks=2 at 15.33.** Insight: on head, GMM chunking at N=2 is
  NOT the efficiency tax the old branch assumed (rung-2: c2 15.52 < c1 15.74), and it buys free
  cross-engine RS hiding. The decouple thesis's "keep GMMs whole" advantage is small/negative at N=2.
- Composition cell running: moe_n_chunks=2 + decouple=4 + barriers + drs (different overlap targets
  may stack) -> vs 15.33.
- STRATEGIC: remaining big prize = INPUT side (token-AG ~720ms/step exposed in every config) =
  rungs 9-11, plus the SC-block slide. Combine-side machinery (correct, flag-gated) ready to compose.

### Rung 9 v1 verdict [2026-07-03]: CORRECTNESS PASS / perf fail (functional-merge tax)
- Full-stack + decouple_dispatch_chunks=4: loss digit-exact (12.270/9.324) — input-side pipeline math
  RIGHT on hardware first try. **19.1s** (+3.9 vs 15.23): the per-chunk jnp.where(mask, gathered, buf)
  fallback = N x full-buffer (~14GB) merges. Predicted cost of the correctness-first fallback.
- Iteration dispatched (rung9b): true in-place disjoint scatter-writes (aliasing chain / SC
  scatter-write pattern), O(chunk_rows) per chunk; addressing emulation required before shipping.

### Rung 9b [2026-07-03] — in-place blocker + resolution
- AOT temps CORRECTION: N-scaling is FLAT (off 106.9 / N2 106.8 / N4 104.3 GB — XLA aliases buf).
  The 19.1s regression is pure TRAFFIC+SC-compute (N x full-buffer gather AND 3x-buffer where-merge
  per chunk), invisible to the peak-memory instrument.
- VERIFIED API BLOCKER: pl.kernel (SC subcore API) lacks input_output_aliases (only pl.pallas_call
  has it, which lacks mesh) — the O(chunk_rows) compaction-write needs kernel-side aliasing.
- DECISION (user pre-authorized "don't wait"): build the conditional-DMA SC scatter/accumulate kernel
  (or in-image pl.kernel aliasing plumb, agent's call), numerics-validated STANDALONE ON v5p SSH
  (the playbook's own validation rig) before any image; cluster loss-digit-match as backstop.

### Rung 9b cluster iterations [2026-07-03]
- attempt 1: scoped-vmem ceiling (wrapper consumes the 64KiB reserve; fix = limit 65472 — same as the
  trace-flag gotcha, now a pattern: NEW SC-side machinery costs the reserve).
- attempt 2 (65472): JAX ValueError — wrapper's jit'd _run RETURNS the mutable Ref (forbidden across
  jit). v5p standalone missed it (no outer-jit embedding). Fix in flight (value-return + verify HLO
  still shows donation not copy); jit-embedding case added to the standalone test.
- attempt 3 (r9c): Ref leak moved to the custom_vjp boundary (result[12]). Fix generalized: freeze
  ref->value inside _chunked_dispatch before ANY return; standalone harness upgraded to the model's
  full composition scan(custom_vjp(core_map)) so boundary-crossing bugs are caught pre-cluster.

### Rung 9 in-place: FRAMEWORK WALL [2026-07-03] — parked with named routes
- Definitive (4 variants root-caused): Pallas SC API cannot express in-place accumulate — new_ref(input)
  = Ref-as-jit-output (forbidden across jit/custom_vjp/scan); pl.kernel lacks input_output_aliases
  (only pl.pallas_call has it; no SC mesh there). Kernel itself VALIDATED on v5p (diff=0, 0 copies,
  temp flat) — docstring-documented as reference (ragged_gather_reduce_accumulate).
- Reverted dispatch to the where-merge (correct, default-off, known Nx-traffic tax). rung9d = safe state.
- ROUTES for the user's call: (A) in-image JAX patch plumbing input_output_aliases through pl.kernel
  (surgical, we control the image; also a legit upstream JAX feature request); (B) compaction-first
  redesign — per-chunk AG + compaction-gather to SMALL [chunk_rows,H] outputs (pipelined), ONE
  deferred full-buffer gather at the end (functional, no refs; costs ~2x SC dispatch pass ~+3.7ms/layer
  vs ~6ms AG prize — marginal-positive, needs measurement); (C) park input side, all value from
  rungs 8/10/11 + combine side.
- INPUT-SIDE STATUS: correctness fully proven on hardware; perf pending one of the routes.

### EP=8 ladder [2026-07-03]
- OPERATOR ERROR (mine): e8a's sed rename didn't match -> EP8 config ran AS ds-v3-r8final-0703,
  clobbering that GCS log dir (r8final numbers preserved in this log; its profile was never pulled).
  Naming now verified pre-apply.
- **e8a verdict (EP8/FSDP64, chunks=2+manbwd+groups): OOM at compile — 103.63G vs 94.74G** (+~14G vs
  EP4's 89.9). The user-predicted "remat adjustment" is required at EP=8.
- e8aoo launched: + optimizer_memory_host_offload=True (frees ~15.8G device-side, known +0.39s tax
  at EP4). If it fits: EP8 reference number, then e8b-oo (+ decouple4 + drs) tests lever scaling.
- EP8 memory dial ledger (all OOM so far): chunks=2 baseline **103.63G** | +opt-offload 103.43G
  (INERT — overage is temps, not args) | chunks=4 101.94G (chunking shaves only 1.7G — x_sorted is
  NOT the EP8 driver; suspect the 2x gathered-token buffer + decoder-input stack) | trying
  decoder_layer_input=offload (the ~13.6G 58-layer input stack; arithmetic closes the 7.2G gap).
- decoder_layer_input=offload UNLOCKS EP8: e8adlo (chunks=2+manbwd) = **16.77s**, loss 9.325 healthy.
- e8bdlo (+decouple4+drs) = **17.26s** — the levers INVERT at EP8 (+0.49 vs −0.10 at EP4; suspects:
  TC-RS 2x bytes over 2x ring + PCIe staging on TC + latency-bound small chunk messages).
- **EP8 CLOSED: EP4 x 15.23-stack wins by ≥1.5s under no-rbf on head.** The old "mesh null" result
  does not survive the no-rbf constraint + head memory layout. Remat adjustment ledger preserved above.

### #2 answered: remat-extension +0.44 attribution [2026-07-03, profile compare r8comp vs r8final]
- (r8final profile SURVIVED the GCS clobber — EP8 run died at compile, never reached profiler steps.)
- The +439ms/step is EXPOSURE, not lane work (TC busy +57ms, SC −7ms): all-gather count +1866
  (8750->10616), AG lane +710ms — the flat bwd's PER-CHUNK cotangent AGs (more, smaller, 462 vs
  493 GB/s, uncovered in the bwd).
- FIX QUEUED (rung-8 iteration): g_out is contiguous -> replace N per-chunk AGs with ONE tiled AG +
  in-register reorder == identical cotangent at unchunked-AG cost. Expected: 15.67 -> ~15.2x, making
  the remat extension free-or-better.
- ROUTE B dispatched (rung9e): compaction-first dispatch — per-chunk AG + compacted gather (small
  outputs, pipelined) + ONE deferred full-buffer gather (safe call class). Cost ~2x dispatch SC pass
  (+3.7ms/layer) vs ~5-6ms/layer AG hidden.

### Remat/offload study [2026-07-03, analyst, receipts in agent report] — THE ROADMAP REDRAWN
- bwd = 69% of step (10.45s = 58x180ms); recompute only ~19% of bwd (2.0s); true grads 5.5s;
  exposed collectives 2.53s/step (43.6ms/layer).
- RANKED: (1) **splash O+LSE host-offload = -630ms/step gross** (10.9ms/layer recompute deleted;
  545MB/layer restore = 3GB/s vs 95%-idle host-DMA lane) — the deferred moe_splash_host_offload port,
  NOW TOP PRIORITY; (2) sort-indices+router-probs device-save (0.4GB, -1-2ms/layer); (3) marginal:
  MoE-input save (235MB/layer, skips o-proj recompute 3.5ms); (4) DO NOT convert gmm/QKV/token-AG
  recomputes (restore 3-8x worse — current manbwd already optimal there); (5) next after remat:
  bwd weight re-gather ~14ms/layer = splash_wag-option-B territory (old branch: -0.82s bwd).
- PATH TO TARGET: 15.23 - 0.63 (splash offload) - ~0.5 (wag bwd) ~= 14.0-14.1 = the ledger's number,
  via two PROVEN old-branch ports.

### Rung 9e (Route B compaction-first dispatch) verdict [2026-07-03] — CORRECTNESS PASS / perf FAIL
- r9e = canonical 15.23 stack + decouple_dispatch_chunks=4 (compaction-first, image rung9e-db413add5):
  loss digit-exact (12.270 -> 9.324), steady-state **15.83s = +0.60 vs 15.23 control**. The ~2x SC
  dispatch-pass tax swamps the ~5-6ms/layer AG prize. INPUT-side dispatch chunking now 0-for-3 on perf
  (where-merge +3.9, in-place = framework wall, compaction-first +0.6). Input side PARKED unless a new
  mechanism appears; flag stays default-off.

### splash O+LSE host-offload first hardware contact [2026-07-03] — CORRECTNESS PASS / perf inverted
- splashoff = canonical stack + moe_splash_host_offload=True (image splashoff-99743694e): loss
  digit-exact 9.324, **15.47s = +0.24 vs 15.23** — projection was −0.63. Offload correct, not free.
- Iteration in flight: ds-v3-splashoff-pcie1-0703 — single-knob A/B, --xla_tpu_pcie_bandwidth_multiplier
  0.03 -> 1.0 (same image/flags). Hypothesis: LHS cost model at 3% PCIe BW refuses to schedule the
  545MB/layer restores under compute -> restores serialize on the bwd critical path.

### splashoff pcie1 A/B [2026-07-03] — NULL. PCIe cost-model knob is NOT the blocker
- pcie_bandwidth_multiplier 0.03 -> 1.0 (single-knob A/B, same image): **15.49s** vs splashoff 15.47
  (delta +0.02 = noise), loss digit-exact 9.324. LHS scheduler PCIe modeling exonerated (this knob).
- splash-offload +0.24 regression stands UNATTRIBUTED -> profile compare splashoff vs r8comp
  (both profiled steps 6-8) is the next instrument.

### splashoff +0.24 ATTRIBUTED [2026-07-03, profile compare r8comp vs splashoff, xla_shell receipts]
- **The mechanism WORKED: TC compute −753ms/step (measured; beats the −630 projection). The regression
  is lost overlap cover, not the offload mechanism.**
- Per-step lane deltas (r8comp 15.23 -> splashoff 15.47): TC lane −605ms (compute −753, vpu +77,
  relayout +70) | SC lane +113ms | Host-DMA busy 702ms->1.19s | exposed collective 21.0%->23.9%
  (+~0.5s) | exposed offload 4.4%->5.9% (+~0.24s). Best-overlap ceiling IMPROVED 12.01->11.41s.
- Smoking gun: NEW exposed bwd all-gather.626 — ~2GB/layer, 228 firings (57 layers x 4 iters),
  9.4ms/iter at 115 GB/s = ~0.48s/step. The deleted splash recompute (10.9ms/layer) was the compute
  COVER for this ~9.4ms/layer bwd weight re-gather; classic recompute-deletion exposure trap
  (same class as the r8final +0.44 AG exposure). Plus structural churn: AG ops 48->74 distinct,
  AG bytes 20.9->33.8GB, RS achieved BW 304->176 GB/s.
- Arithmetic closes: −0.75 (compute) + 0.48 (AG exposure) + 0.24 (offload copies) + 0.15 (vpu/relayout)
  + SC/misc ≈ +0.24 net.
- RECOVERY PATH (win is real if exposure is re-covered; ceiling says ≥0.6s available): (1) scheduling
  cover for the exposed bwd AG — scheduling-group annotation or reorder (moe_weight_ag_scheduling_group
  class fix); (2) prefetch host restores layers-ahead so copies hide (Host-DMA lane is 92% idle);
  (3) this is also exactly splash_wag-option-B territory (fuse weight AG into splash bwd kernel) —
  but option B's old form hid AG under the recompute, which no longer exists; needs redesign vs dkv.
- pcie1 null now explained: the knob tunes the cost model, but the problem is a MISSING overlap
  target, not a mispriced one.

### Rung 8b VERDICT [2026-07-03] — WIN, new best 15.22, remat extension now FREE
- Fix (commit 522d59086, image rung8b-522d59086): flat bwd's N per-chunk cotangent AGs -> ONE tiled
  AG, row-reorder folded into the gather's token indices (zero extra buffers; a data-transpose variant
  was tried and REJECTED on a measured +27MB temp). Receipts: bit-exact grads (EP 4/8 x N 1/2/4/8 x
  f32/bf16, exact equality), HLO A/B 4x bf16[32768,7168] AG -> 1x bf16[131072,7168], peak temp flat.
- Cluster (ds-v3-r8b-0703 = canonical stack + moe_chunked_combine_in_remat=True): **15.22s**
  steady-state, loss digit-exact 9.324. vs r8final 15.67 (-0.45 recovered) and vs canonical 15.23
  (-0.01 = tie/hair better). **NEW REFERENCE STACK: r8b config** — same speed as canonical but
  memory-flat bwd (composes with razor-thin 128-chip gates).
- NEXT: splashoff recovery on top of r8b (scheduling cover for the exposed bwd AG + restore prefetch;
  ceiling receipt says ≥0.6s available).

### CORRECTION + recovery-agent receipts [2026-07-03] — all-gather.626 is the COMBINE COTANGENT AG
- **Prior entry corrected: .626 is NOT a weight re-gather.** HLO source map: it is the _drs_bwd
  transpose of the direct-RS combine (all_gather of g_out) in moe.py; weight re-gathers are
  .523/.525/.527/.535. Cover story unchanged (splash recompute was its cover), tensor identity fixed.
- SC-queue receipts (user's catch, verified in scheduled HLO): combine AG + weight AGs BOTH ride the
  SC offload queue (async_execution_thread="sparsecore") -> cannot overlap each other; host O/LSE
  restore = async dynamic-slice-start from pinned-host, NOT on the SC queue -> restore ∥ SC-AG feasible.
- Naive frontend scheduling-group fix: AOT-NEGATIVE (un-fuses the merged AG, singleton group — the
  tag does not survive onto the MSA-generated restore copy, +2.15GB peak temp). Committed gated
  default-off (2dd506daa); restore prefetch is an MSA cost-model lever, not a frontend group.
- BRANCH GOTCHA: the stagger/barrier machinery (_moe_staggered/_gather_wo/moe_handwritten_bwd_barrier)
  exists only on the decouple-combine-rs-chunks lineage (main checkout), NOT on rung6-sliced (the
  image lineage, 924-line deepseek.py). Using it = a port task.
- DISPATCHED: ds-v3-r8bso-0703 = r8b 15.22 stack + moe_splash_host_offload=True (image
  r8bso-2dd506daa, scheduling flag OFF). Rationale: rung 8b restructured exactly the AG that was
  exposed (N per-chunk -> 1 tiled, different schedule neighborhood) -> the splashoff exposure
  arithmetic must be re-measured on the new reference before more scheduling work.

### r8bso [2026-07-03] — splash offload on the r8b stack: exposure UNCHANGED (+0.23)
- ds-v3-r8bso-0703 (r8b 15.22 stack + moe_splash_host_offload, image r8bso-2dd506daa, scheduling
  flag OFF): **15.45s**, endpoint loss 9.324. The rung-8b combine restructure did NOT reduce the
  offload exposure (+0.23 vs +0.24 on the old reference). The −0.75s compute win stays locked.
- NUMERICS OBSERVATION (flagged, not blocking): offload lineage prints 9.715 @ step 13 vs 9.714
  non-offload (both offload runs agree across different combine-bwd impls; perplexity digits diverge
  from step 2). Correlates exactly with moe_splash_host_offload. Suspect: offload bwd uses STOCK
  tokamax dkv fed saved LSE vs fused bwd kernel path -> different reduction order. The code's
  "bit-exact" claim is NOT strictly true; endpoint matches. Needs a proper A/B if it ever matters.
- SCREENING DISPATCHED: ds-v3-r8bsofl-0703 = r8bso + 4 untested scheduler flags from the playbook
  (host_transfer_overlap_limit=128, max_concurrent_host_send_recv=128,
  lhs_prioritize_async_depth_over_stall=true, ag_backward_pipelining=true, lhs rerun=2).
  All-at-once screen; bisect only if it moves. pcie multiplier already measured-dead.

### FLAG SCREEN HIT [2026-07-03] — r8bsofl 14.65s, NEW OVERALL BEST (−0.57 vs r8b 15.22)
- ds-v3-r8bsofl-0703 = r8bso (offload) + 4 untested playbook flags (host_transfer_overlap_limit=128,
  max_concurrent_host_send_recv=128, lhs_prioritize_async_depth_over_stall=true,
  ag_backward_pipelining=true, latency_hiding_scheduler_rerun=2): **14.65s** steady-state
  (14.639-14.654), endpoint 9.324. −0.80 vs r8bso 15.45. The splash-offload −0.75s compute win
  REALIZED + more. (Playbook source vindicated on these; pcie multiplier remains its one dead claim.)
- Numerics: step-13 prints 9.714 (matches non-offload lineage), perplexity low digits differ from
  both prior lineages — scheduler flags restructure collective/reduction order, low-bit drift,
  endpoint exact. Same class as the offload dkv observation.
- ISOLATION RUNNING: ds-v3-r8bfl-0703 = r8b stack + same flags, NO offload — do the flags need the
  offload or lift the base too? Then bisect flags on whichever config wins.
- Profile captured (steps 6-8) at gs://sivaibhav-exp/maxtext/ds-v3-r8bsofl-0703 for attribution.

### Flag isolation 2x2 complete [2026-07-03] — flags are offload-ONLY synergy
- ds-v3-r8bfl-0703 (r8b + 4 flags, NO offload): **15.23s** = null vs r8b 15.22. Full 2x2:
  base 15.22 | base+flags 15.23 | offload 15.45 | offload+flags **14.65**. Interaction −0.81:
  the flags' whole value is unlocking offload exposure; base schedule gains nothing
  (ag_backward_pipelining alone does NOT help the non-offload AG exposure — note for the
  remaining 3.43s exposed-collective prize: flags won't touch it without new offload-class work).
- Lane receipt (r8bso vs r8bsofl, same-image single-diff): exposed offload 0.91->0.41 (−0.50),
  exposed collective 3.70->3.43 (−0.27), TC compute unchanged 8.27 — sums to the −0.80.
- BISECT cell H launched: ds-v3-r8bsoflh-0703 = offload + host_transfer_overlap_limit=128 +
  max_concurrent_host_send_recv=128 only. Prediction from lanes: ~−0.5 (restore hiding).
  Cell S (async_depth + ag_backward_pipelining + rerun) next if H doesn't explain the full −0.8.

### Bisect cell H [2026-07-03] — NULL. Host-transfer pair alone does nothing
- ds-v3-r8bsoflh-0703 (offload + host_transfer_overlap_limit=128 + max_concurrent_host_send_recv=128
  ONLY): **15.45s** = r8bso exactly. Lane-based prediction (~−0.5 from restore hiding) WRONG —
  the restore exposure does not fall to transfer-concurrency limits alone.
- Cell S launched: ds-v3-r8bsofls-0703 = offload + scheduler trio (lhs_prioritize_async_depth_over_stall
  + ag_backward_pipelining + latency_hiding_scheduler_rerun=2). If S=14.65 the pair is droppable;
  if intermediate, it's an interaction (scheduler wants to prefetch deeper but the transfer limit caps
  it -> both needed).

### Bisect cell S [2026-07-03] — SCHEDULER TRIO CARRIES EVERYTHING: 14.64s
- ds-v3-r8bsofls-0703 (offload + lhs_prioritize_async_depth_over_stall + ag_backward_pipelining +
  latency_hiding_scheduler_rerun=2, NO host-transfer pair): **14.64s** (14.640-14.649) = the full
  5-flag stack. Host-transfer pair fully droppable (cell H null + cell S complete = clean split).
- Final cell launched: ds-v3-r8bsoflag-0703 = offload + ag_backward_pipelining ALONE.
- Canonical flag candidate: offload + trio (or fewer, pending the last cell).

### Bisect: ag_backward_pipelining alone NULL [2026-07-03]
- ds-v3-r8bsoflag-0703 (offload + ag_backward_pipelining only): **15.45s** = no effect alone.
  Both single-lever priors (host-transfer pair, ag-pipelining) now measured-wrong; the win lives in
  lhs_prioritize_async_depth_over_stall and/or latency_hiding_scheduler_rerun (or a trio interaction).
- Cell launched: ds-v3-r8bsofladr-0703 = offload + async_depth + rerun (no ag). If 14.64 -> ag
  droppable, one more split (async_depth alone) finishes; if null -> trio interaction, freeze trio.

### Bisect FINAL [2026-07-03] — carrier is async_depth + rerun; ag-pipelining droppable
- ds-v3-r8bsofladr-0703 (offload + lhs_prioritize_async_depth_over_stall + scheduler_rerun=2, no ag):
  **14.64s** = full effect. Complete tree: none/pair-H/ag-alone all 15.45 | trio 14.64 | adr pair 14.64
  | all five 14.65. MINIMAL CANONICAL FLAG SET = offload + async_depth_over_stall + rerun=2.
  (Split of adr into singles deferred — diminishing returns; do it if a flag needs upstreaming.)
- QUEUED (user): ds-v3-r8bsosch-0703 = adr stack + use_splash_scheduler=True (types.py:708 ->
  dkv kernel use_experimental_scheduler; only unmeasured in-kernel lever for the 18ms x 58 dkv).

### use_splash_scheduler A/B [2026-07-03] — REGRESSION, retired
- ds-v3-r8bsosch-0703 (adr stack + use_splash_scheduler=True): **14.80-14.82s** vs 14.64 control
  (+0.16), loss endpoint exact. Experimental dkv scheduler is WORSE at sa_block_*_dkv=2048 on this
  shape. dkv in-kernel lever list now empty (kernel at FLOP floor); its role = overlap cover.
- Final bisect split launched: ds-v3-r8bsoad-0703 = offload + lhs_prioritize_async_depth_over_stall
  ALONE (is the canonical set a single flag + rerun droppable?).

### BISECT RESOLVED: single-flag carrier [2026-07-03] — lhs_prioritize_async_depth_over_stall
- ds-v3-r8bsoad-0703 (offload + lhs_prioritize_async_depth_over_stall=true ONLY): **14.64-14.66s**
  = full effect. rerun=2 also droppable. THE ENTIRE −0.80 = one LHS flag, active only when
  moe_splash_host_offload gives it async work + freed compute to exploit (base-stack null).
- **CANONICAL CONFIG (14.64s BEST): r8b stack + moe_splash_host_offload=True +
  --xla_lhs_prioritize_async_depth_over_stall=true** (manifest: r8bsoad_manifest.yaml, image
  r8bso-2dd506daa). Full retired-lever list this session: pcie multiplier (both values),
  host-transfer limits, ag_backward_pipelining, scheduler rerun, use_splash_scheduler (+0.16 regr),
  frontend scheduling groups, input-side dispatch chunking.

### Causal-mask block-granularity finding [2026-07-03] + dkv block A/B launched
- dkv kernel IS causal (CausalMask, attention_op.py:1279; iota operand = in-block masking for
  diagonal blocks) BUT at sa_block_*_dkv=2048, seq 4096 -> 2x2 block grid: causal skips only 1 of 4
  blocks; the 2 diagonal blocks burn full FLOPs half-masked -> kernel executes ~75% of full-mask
  work vs ~50% ideal (derived). Theoretical max recovery ~4.5ms/layer ~ 0.26s/step; fwd kernel
  (block 2048) has the same 75% factor.
- A/B launched: ds-v3-r8bsodkv1k-0703 = canonical 14.64 + sa_block_{q,kv,kv_compute}_dkv=1024
  (4x4 grid -> 62.5% of full work; fwd blocks untouched). Tradeoff vs per-block overhead/MXU eff
  at smaller tiles = why measured not assumed. Loss must stay digit-exact (mask math unchanged).

### dkv block-size A/B [2026-07-03] — 1024 REGRESSES (+0.09); 512 queued
- ds-v3-r8bsodkv1k-0703 (canonical + sa_block_*_dkv=1024): **14.73-14.75s** vs 14.64 control, loss
  digit-exact. The 12.5-pt FLOP saving from finer causal capture (75%->62.5% of full-mask work) is
  MORE than eaten by small-tile overhead. 2048 stands as the dkv sweet spot pending the 512 point.
- ds-v3-r8bsodkv512-0703 (sa_block_*_dkv=512, 8x8 grid -> 56%) queued behind it per user request —
  closes the curve; expectation (derived from the 1024 trend): worse. fwd-block variant will be
  judged by the same curve.

### dkv block 512 [2026-07-03] — 15.88s, cliff confirms the compute-tile confound
- ds-v3-r8bsodkv512-0703: **15.88s** (+1.24 vs 14.64), loss digit-exact. Curve 2048/1024/512 =
  14.64/14.74/15.88 — nonlinear MXU-efficiency cliff. USER-IDENTIFIED CONFOUND: block_kv_dkv_compute
  is capped by block_kv_dkv, so symmetric shrink measures the inner-matmul tile tax, NOT causal
  capture. Clean experiment (decouple grid from compute tile) in progress by user w/ splash agent;
  prize if separable: up to ~0.26s/step dkv + similar factor on fwd kernel. Arm closed on my side.

### Device-saves LANDED (agent) + validation cells dispatched [2026-07-03]
- Commits on rung6-sliced: 52d5b000c (CPU ring-path enabler), 1e564485f (moe_save_block_input),
  c22630592 (moe_save_sort_indices). Image devsaves-c22630592.
- Agent receipts: sort-indices grads BIT-EXACT + memory FLAT (−57KB) + 4 recompute argsorts/layer
  GONE from bwd HLO; block-input mechanism sound (saved==replayed, probed 0.0) but grads wobble
  ~1e-10 f32 (XLA re-fusion class) and peak 73.19->87.63 GiB = **0.61 GiB under the gate**.
- BONUS BUG: nnx bridge checkpoint-skip gate used `is` vs a dynamic subclass -> could silently
  not fire; fixed to isinstance (1e564485f).
- Cells: ds-v3-r8bsosi-0703 (canonical+sort_indices; acceptance DIGIT-EXACT) then ds-v3-r8bsoboth-0703
  (+block_input; low-bit drift allowed, endpoint must match; compile-OOM watch at 0.61 GiB margin).

### Device-saves cluster verdict [2026-07-03] — BOTH RETIRED (default-off)
- sort_indices: 14.76 (+0.11, digit-exact) — argsorts were already hidden; residual traffic costs
  more than the deleted compute. block_input marginal (both-cell 14.73): ~−0.03 for 14.44 GiB =
  all headroom for noise. Study items #2/#3 measured-dead on hardware; code stays (gated off).
- Compile fit at 0.61 GiB margin confirmed on hardware (both-cell ran). Headroom RESERVED for
  stagger-port restore-prefetch depth instead. Remat-save track CLOSED; remaining roadmap:
  stagger/prefetch port (dispatched), user's splash block benchmark, wag-bwd port.

### moe_bwd_xlayer_prefetch LANDED (agent) + cluster A/B dispatched [2026-07-03]
- Commit 85f6ad2d3 (image xprefetch-85f6ad2d3): reverse-scan prefetch of the NEXT backward layer's
  up-proj (wi_0/wi_1) FSDP AG into the ~18ms dkv window. NNX-decoder path (spec correction: prod is
  NNX not linen); prev-layer W01 slice passed as scan input (no param lift -> grad tree identical);
  handed down via cotangent of an identity dummy carry (reverse-scan data channel); consumed by
  swap_gather_w01 custom_vjp (bwd = same psum_scatter); top layer lax.cond fallback. wo + combine
  cotangent AG NOT touched (subset of the 3.43s exposed mass).
- Receipts (agent, measured): CPU loss+grad BIT-EXACT 0.0 (NNX, real+random); flag-off byte-identical;
  SCHEDULE receipt = producer AG call-start@138 < dkv@804 < AG call-done@929 (real overlap, in-flight
  across dkv). MEMORY RISK: +22.29 GB compile-time temp @tpu7x-512 (86.72->109.01, host_temp flat) --
  producer AG live across whole bwd body; pessimistic instrument but gate-tight -> cluster confirms fit.
- ds-v3-r8bsoxpf-0703 dispatched (canonical 14.64 + prefetch). Watch: compile-OOM first, then perf vs
  14.64 + loss digit-exact. If OOM -> w0-only mitigation (halve the carry).

### moe_bwd_xlayer_prefetch VERDICT [2026-07-04] — DOUBLE FAIL, parked default-off
- ds-v3-r8bsoxpf-0703 (canonical 14.64 + prefetch, image xprefetch-85f6ad2d3):
  (a) NUMERICS DIVERGE on hardware: step19 loss 9.313 vs canonical 9.324 (systematic, whole curve
      lower: s11 9.979 vs 9.987, s13 9.704 vs 9.714; ppl 11077 vs 11207). 9.324 endpoint was
      digit-identical across ~15 runs today -> real divergence. CPU test said grad diff 0.0 ->
      HARDWARE-ONLY, mini-config missed it. Suspect: bf16 reduce-order in swap_gather_w01 psum_scatter
      via the reverse-scan cotangent channel (f32-exact CPU / bf16-divergent TPU), OR a real grad bug
      at a boundary the 8L mini-scan doesn't hit (top-layer lax.cond fallback / 58L carry). FAILS the
      digit-exact bar.
  (b) PERF REGRESSION: 15.15s = +0.51 vs 14.64. Compiled+ran (the +22GB fit at runtime, no OOM), so
      memory was NOT the blocker -> w0-only mitigation is MOOT. The producer AG held live across the
      whole bwd body contends with the layer's own weight-grad RSs on SC/ICI instead of hiding free;
      the schedule receipt showed emission-before-dkv but not free overlap.
- Flag parked default-off; code retained. Cross-iteration prefetch (this shape) does NOT convert the
  exposed up-proj-AG mass -> the free-overlap thesis for the dkv window is UNPROVEN. Reassess before
  any more work on this route: numerics root-cause (reduce-order=tolerable vs grad-bug=must-fix) +
  the queue-contention model (why the in-flight AG steals from the layer's own collectives).

### chunk=1 backward hypothesis — cheap flag-only probe first [2026-07-04]
- User thesis (conceded valid; my roofline argued the wrong axis): chunk pipeline hides SC
  dispatch-gather + combine gather-reduce under next-chunk GMM; in the BACKWARD the big tgmm/dkv
  blocks may already provide that cross-engine cover WITHOUT the chunk split -> chunk=1 bwd drops
  per-chunk overhead (2x launches, concat, sum, smaller tiles) while keeping the hiding. Lower risk
  than prefetch.
- Cheap isolation BEFORE the decouple code: ds-v3-r8bsoc1-0704 = canonical 14.64 + moe_n_chunks=1
  (SYMMETRIC fwd+bwd=1, flag-only, no build). decouple_combine_rs_chunks=4 + chunked_combine_in_remat
  kept (orthogonal). Plan: profile-compare the BACKWARD lane vs 14.64 chunk=2. If bwd faster at c1
  (even as fwd pays the rung-2 tax) -> GREEN LIGHT to build decoupled fwd=2/bwd=1. If bwd flat -> shelve.
- NOTE: current-stack chunk=1 is itself a fresh datapoint — rung-2's c1 15.74 predates splash-offload
  + LHS-flag + rung-8b (un-chunked combine AG), so the c1/c2 gap may have moved.

### chunk=1 backward-lane verdict [2026-07-04] — user hunch has REAL signal, masked by offload cover
- Symmetric moe_n_chunks=1 (ds-v3-r8bsoc1-0704): 14.76s (+0.11 vs 14.64), loss 9.325 (~digit-exact).
- BACKWARD lane compare (abs ms/step, chunk=1 - chunk=2 @ their own step times):
  bwd compute 4.50 vs 4.89 = **-0.39s** (chunk=1 lower -- overhead/launch, NOT MXU; my roofline
  objection was the wrong axis, user right) | bwd exposed collective 2.20 vs 2.33 = **-0.13s**
  (tgmm/dkv cover the SC gathers fine WITHOUT chunk pipelining -- user thesis validated) |
  bwd OFFLOAD 0.78 vs 0.18 = **+0.60s** (host O/LSE restore loses its cover at chunk=1 -- the killer)
  | bwd total 9.11 vs 9.05 = +0.06 flat.
- READ: chunk=1 bwd wins -0.52s on compute+collective, masked by a +0.60s restore-cover regression
  that looks RECOVERABLE (same "give the async op cover" the LHS flag solved once). Decoupled
  fwd=2/bwd=1 worth building IF bundled with re-covering the restore at chunk=1. Compute+collective
  win now MEASURED not hypothetical.

### all-gather.626 vs .527 HLO trace [2026-07-04] — parallelism blocked by SC-core COLLISION, not dataflow
- .626 = chunked-combine-rs-bwd/all_gather (combine cotangent AG). Operands = cotangent slices
  (fusion.1436<-fusion.1435, src jvp/shard_map) = g_out = layer output cotangent -> roots in prev
  layer dx (dkv). CONFIRMED dx-locked, can't move. bf16[32768,7168], SUBCORE_TYPE_SCS, core_ids["0"].
- .527.cloned.1 = jvp/shard_map/all_gather, bf16[1,64,7168,2048] = EXPERT WEIGHT (E64/D7168/F2048),
  fed by weight data-format-call = PARAM-dependent, can start early. SUBCORE_TYPE_SCS, core_ids["0"].
- **BOTH offloaded to SC scalar-subcore 0 -> physically serialize -> both exposed.** BUT census of 74
  AG offloads: 44 on core 0, 30 on core 1 -> TWO SCs in use, compiler load-balances. The exposed pair
  just collided on core 0. Parallelism PHYSICALLY AVAILABLE.
- TESTABLE LEVERS: (a) --xla_tpu_use_single_sparse_core_for_all_gather_offload=false -> each AG shards
  across BOTH SCs -> ~2x faster per AG -> attacks the 9.4ms .626 exposure directly (flag flip);
  (b) influence assignment so the exposed combine-AG + weight-AG land on different cores.

### multi-SC AG offload = NEW BEST 14.50 [2026-07-04] — user's .626||.527 parallelism call, CONFIRMED
- ds-v3-r8bsomsc-0704 = canonical + xla_tpu_use_single_sparse_core_for_all_gather_offload=FALSE:
  **14.50s (14.493-14.511)** vs 14.64 (−0.14), loss digit-exact 9.324. NEW CANONICAL
  (canonical_best_14.50_manifest.yaml). 0.04s off the 14.54 all-time record on the clean no-rbf lineage.
- MECHANISM (measured, not assumed): .626 combine AG UNCHANGED (9.384ms/111 GB/s, was 9.413/105) ->
  the win is NOT .626 speeding up. It's the .527-class weight AG now running on the OTHER SC in
  PARALLEL with .626 instead of serializing behind it on core 0 (~5ms weight-AG hidden). Exactly the
  user's ".626 || .527 should run in parallel" prediction from the HLO core-collision trace.
- CAVEAT: early log shows scoped-VMEM INVALID_ARGUMENT (67108864 req vs 67043328 max, the 64KiB
  reserve gotcha) -- non-fatal, XLA clamped, ran clean + exact loss. Pin xla_tpu_scoped_vmem_limit_kib
  =65472 in canonical to silence + confirm no tile lost.
- REMAINING on .626: still 9.4ms/111 GB/s exposed tail (it did NOT shard across cores -- merged
  3-tuple AG). Making .626 ITSELF faster is the next lever on it (separate from the parallelism win).

### vmem-pin confirm [2026-07-04] — 14.50 HOLDS clean
- ds-v3-r8bsomscv-0704 = 14.50 best + xla_tpu_scoped_vmem_limit_kib=65472: **14.50s** (14.496-14.546),
  loss digit-exact 9.324, scoped-VMEM warning GONE (0). Warning was a benign clamp (no tile lost).
  CANONICAL = this pinned version (canonical_best_14.50_manifest.yaml updated). 0.04s off 14.54 record.

### .626 combine-AG speedup investigation [2026-07-04, agent, AOT receipts] — un-merge REFUTED
- CORRECTION to prior claim: the 3-tuple merge is NOT why .626 stays single-SC. AOT (flag ON vs OFF,
  tpu7x-128): un-merged form STILL carries use_single_sparse_core:true, lands whole on 1 SC. The
  merge is a 61-layer scale artifact (AllGatherCombiner), absent at 8 layers.
- ROOT CAUSE (measured): single-SC is a PER-OP heuristic keyed on .626 being a dim-0 gather over the
  CONSECUTIVE EP ring {0,1,2,3}; its consumer is a co-located SC sc_ragged_gather kernel. The weight
  AGs that DO split across both SCs are dim-1 FSDP gathers over STRIDED cores {0,4,8,...}. 111 GB/s =
  topology (4-core EP ring, dim-0), not merge. Code: ragged_sort.py:774 all_gather(g_out, ep, tiled).
- The 3-tuple = out0/out1 = the two combine cotangent gathers (bf16[32768,7168]) + out2 = a FORWARD
  RECOMPUTE gather (bf16[16,2048,7168]) that XLA bundled in and kept on the bwd crit path.
- VERDICT: (a) un-merge REJECTED as bw fix (AOT: still single-SC, no per-AG gain). (b) REAL lever =
  force tensor_split_factor=2 for dim-0/EP gathers (gated off) OR move this AG to the TC path -- needs
  flag hunt / code, no receipt yet. (c) decouple_combine_rs_chunks REJECTED (rung-8 = 1 AG for all N).
- CHEAP runtime probe launching: xla_tpu_all_gather_combine_threshold_bytes lowered to un-merge
  (bit-identical) -> tests if the 2 combine gathers land on cores 0+1 (parallel) + drop out2 off crit
  path. Long-shot minor; closes the un-merge question with a cluster datapoint.

### un-merge probe [2026-07-04] — INVALID FLAG, branch closed
- ds-v3-r8bsonomerge-0704 FAILED: "Unknown command line flag
  'xla_tpu_all_gather_combine_threshold_bytes'" — the agent-suggested combiner flag does NOT exist in
  this libtpu. (My lapse: ran it without verifying the flag name.) Jobset deleted.
- Un-merge branch CLOSED: AOT already refuted the bandwidth premise (un-merged form stays single-SC);
  the only open sub-question (2 combine gathers on cores 0+1) was a thin maybe, not worth a flag hunt.
- .626 REAL lever remains: force tensor_split_factor=2 for dim-0/EP-ring gathers (needs the correct
  gating flag, unknown) OR move the combine-bwd AG to the TC path (code change in _drs_bwd). Parked.

### moe_bwd_n_chunks LANDED (agent) + cluster A/B dispatched [2026-07-04]
- Commit 6011e1073 on moe-bwd-n-chunks-1 (image bwdnc1-6011e1073): fwd stays chunk=2, bwd MoE
  recompute runs 1 concatenated full-m pass. Gates: AOT both compile under gate (on 86.71 < off 86.83
  GiB, LOWER); HLO tgmm m 2xhalf->1xfull, tgmm count 8->4, fwd gmm unchanged. CPU loss BIT-EXACT
  (from unchanged fwd), grads differ by DESIGNED reduce-order (2.3e-6 random / 1.4e-12 real f32).
- NUMERICS WATCH: grad reduce-order compounds over steps -> expect a small loss drift on hardware
  (like prefetch's 9.313 vs 9.324), but here it's a legit algorithmic reduce-order, not a bug.
  Acceptance = judgment call on drift magnitude.
- PERF WATCH: expected −0.52 best (larger GMM re-covers restore) to +0.08 worst (restore-cover
  regression follows bwd=1). ds-v3-bwdnc1-0704 dispatched vs 14.50 canonical. Watch BOTH loss curve
  and the offload/restore lane.

### moe_bwd_n_chunks=1 VERDICT [2026-07-04] — +0.10 regression, restore-cover confirmed; parked
- ds-v3-bwdnc1-0704 (canonical 14.50 + moe_bwd_n_chunks=1): **14.60s (+0.10)**, loss 9.325 (+0.001
  vs 9.324 — the DESIGNED bwd reduce-order, 10x smaller than prefetch's 9.313 divergence, positive,
  within-bar; numerics NOT the concern).
- MECHANISM CONFIRMED (profile compare vs 14.50, backward lane abs ms): bwd compute 4.65->4.34
  = −0.31s (the chunk=1 compute win, real) | bwd OFFLOAD 0.16->0.74 = +0.58s (host O/LSE restore
  lost its cover). Net +0.10. Same restore-cover regression as symmetric chunk=1 (+0.11) -> proven a
  PURE bwd-chunk-1 effect (fewer/bigger blocks = less independent compute for the LHS to hide the
  restore under). The larger single GMM did NOT re-cover it (agent's best-case projection did not hold).
- Flag moe_bwd_n_chunks (commit 6011e1073) correct + gated default-off; PARKED. Rescue path = bundle
  a RESTORE-COVER fix (place the restore under the big GMM / a scheduling hint) -> would capture the
  −0.31..−0.52s. That's the gating follow-up before this lever pays. Profile: bwdnc1-14.60s in dropdown.
- SCOREBOARD UNCHANGED: 14.50 remains best.

### EP=8 + chunk=2 re-measure on 14.50 stack [2026-07-04, user idea]
- Hypothesis (user): EP=8 keeps GMM token count healthy + chunk=2 hides the gather-reduce; memory tricky.
- PRIOR (stale, pre-splash-offload): e8adlo (EP8/FSDP64 chunks=2+manbwd) = 16.77s vs EP4 15.23;
  levers INVERTED at EP8 (e8bdlo +decouple+drs = 17.26); "EP8 closed" verdict predates the 14.50 stack.
- ds-v3-ep8c2-0704 = canonical 14.50 stack + ici_expert_parallelism=8, ici_fsdp_parallelism=64,
  decoder_layer_input=offload (EP8 needs it — FSDP64 holds 2x gathered weight, old EP8 OOM'd 103.6G).
  RISK: decoder_layer_input=offload + moe_splash_host_offload both on host-DMA lane -> may contend.
  Watch: compile-OOM, perf vs 14.50, loss digit-exact, and (profile) whether gather-reduce hides.

### EP=8 OOM [2026-07-04] — tighter than old stack, needs a 2nd host-offload
- ds-v3-ep8c2-0704 (EP8/FSDP64 + decoder_layer_input=offload) OOM at compile: HLO temps 101.45G >
  94.74G (over by 6.7G). CURRENT stack is TIGHTER at EP8 than old (old fit with just
  decoder_layer_input=offload -> e8adlo) — splash-offload restore buffers + multi-SC scratch add
  device pressure. Retry: + optimizer_memory_host_offload=True (frees ~15.8G, +0.39s tax). Now 3
  host-offloads on the DMA lane -> perf host-contention-bound; profile still answers gather-reduce hiding.

### EP=8 + chunk=2 VERDICT [2026-07-04] — MEMORY-INFEASIBLE on 14.50 stack, closed
- 2 OOMs: EP8/FSDP64 + decoder_layer_input=offload = 101.45G; + optimizer_memory_host_offload = 96.70G
  (both > 94.74G gate). Optimizer offload freed only ~4.75G (NOT the 15.8G hoped) -> the EP8 overage is
  HLO TEMPORARIES (FSDP64 2x-gathered weights + token buffers), which optimizer-STATE offload can't touch.
  Closing the last 1.96G needs either chunks=4 (breaks the chunk=2 premise; borderline ~0.26G over) or a
  3rd host-offload (perf-killing, already 2 on the DMA lane).
- STRATEGIC: the gather-reduce hiding the user wanted is ALREADY delivered by EP=4 chunk=2 (= canonical
  14.50 — that IS what chunk=2 does). EP=8 was historically +1.5s (e8adlo 16.77) AND now doesn't fit.
  EP=8 stays CLOSED on head. 14.50 (EP=4 chunk=2) stands as best.

### qk_diag_skip port DISPATCHED [2026-07-04, from splash-agent handoff]
- Handoff: ~/perf-drills/pipelined-dkv/MAXTEXT_HANDOFF.md. Bit-exact splash fused-dkv-bwd opt: skips
  the fully-masked causal-diagonal QK sub-tiles (g×g sub-grid on diagonal blocks only; dv/dk/dq/exp
  stay full-tile -> avoids the MXU cliff my whole-block dkv shrink hit at 14.73/15.88). Branch
  pipelined-dkv-kernel commits cc2c786 (−2.34% dkv, min −3.5%) + da9356d. This is the payoff of the
  user's "decouple grid from compute tile" splash work.
- Preconditions ALL met on canonical 14.50: use_tokamax_splash + sa_use_fused_bwd_kernel + causal flash
  + sa_block_q_dkv==kv_dkv_compute==2048 + bf16 + seq4096/block2048 = N=2 SWEET SPOT (leverage 2/(N+1)).
- tokamax comes from the BASE image (integrate_v2:xlayer-hierrs-nl), not pip -> port = patch in-image
  tokamax splash kernel (Dockerfile overlay) + wire qk_diag_skip/qk_diag_grid through attention_op.py
  create_sa_config (like use_experimental_scheduler) + types.py + base.yml. Dev agent on branch qk-diag-skip.
- Expected: kernel −2-3% MEASURED; end-to-end DERIVED ~−0.04s (−3% of the ~1.05s dkv) -> ~14.46, bit-exact,
  stacks. FWD splash kernel has same waste = follow-up. A/B: loss MUST be digit-identical; qk_diag_grid=4.

### QUEUED: bwd w-ag + token-ag co-schedule group [2026-07-04, user idea]
- OBSERVATION (user, from profile): multi-SC flag (14.50 win) splits MANY collectives across the 2 SCs,
  but NOT the backward w-ag/token-ag pair — they still serialize on one SC (consistent w/ dim-1 FSDP
  gathers splitting while dim-0/EP stay single-SC). Both measured SC-offloaded.
- Hypothesis: co-tag the backward w-ag + token-ag with a SHARED _scheduling_group_id -> nudge the
  scheduler/placer to overlap them (ideally onto different SCs like the pairs that already split).
  User's point: two PURE collectives -> the fuse-serialize scheduling-group gotcha shouldn't fire.
- Dev agent (branch bwd-ag-coschedule) preparing the gated flag moe_bwd_ag_coschedule_group in
  PARALLEL with qk_diag_skip. Gates: flag-off byte-identical; AOT shows both AGs same group id + NOT
  fused/serialized-worse (verify the gotcha didn't fire); loss digit-exact on cluster.
- RUN ORDER (user): A/B qk_diag_skip FIRST, then this. Cluster runs sequential; prep parallel.

### qk_diag_skip A/B LAUNCHED [2026-07-04] — same-image on-vs-off
- Port committed c8aee9d08 (branch qk-diag-skip), image qkdiag-c8aee9d08 (Dockerfile.qkdiag = baseline
  + tokamax splash kernel overlay at /usr/local/lib/python3.12/.../splash_attention_kernel.py).
- AOT gates PASS: flag-off byte-identical HLO (0 non-metadata diff -> overlay inert when off);
  flag-on sub-grid fires (scoped VMEM 44.63M->40.33M, cost-estimate FLOPs identical = same math).
  Hardware self-time not re-run (dev-box TPU busy); bit-exact by construction + measured dQ/dK/dV=0
  in source commits cc2c786/da9356d.
- A/B: ds-v3-qkdiagoff-0704 (qk_diag_skip=False, control on SAME image) vs ds-v3-qkdiagon-0704
  (qk_diag_skip=True qk_diag_grid=4). Same-image to isolate a ~0.04s effect from cross-image noise.
  Both preempt fl28b (maxtext-high). Acceptance: loss digit-exact 9.324 BOTH; on < off = the win.

### bwd w-ag/token-ag co-schedule LANDED (agent) — staged for A/B after qk_diag [2026-07-04]
- Commit d5937c963 (branch bwd-ag-coschedule), image bwdagcosched-d5937c963. Flag
  moe_bwd_ag_coschedule_group: tags backward EP token-AG (all_gather_dim=0, EP groups) with the SAME
  _scheduling_group_id="1" as w0's weight-AG (all_gather_dim=1, FSDP groups).
- GOTCHA RESOLVED: fuse-serialize CANNOT fire — token-AG (EP replica groups) vs w-AG (FSDP replica
  groups) have DIFFERENT replica groups -> combiner legally cannot fuse them (stronger than the user's
  "pure collectives" reasoning). Gates: flag-off byte-identical (only the tag + auto-name shifts);
  numerics loss+grad diff = 0 (random+real). SC core_id placement check = needs cluster (the A/B).
- A/B PLAN (after qk_diag): same-image off vs on (moe_bwd_ag_coschedule_group), image
  bwdagcosched-d5937c963, canonical flags. Acceptance: loss digit-exact; check token-AG now lands on a
  DIFFERENT SC core_id (overlap) + both AG op counts unchanged (not fused).

### qk_diag_skip VERDICT [2026-07-04] — WIN −0.07s clean, bit-exact, KEEPER
- Same-image A/B (image qkdiag-c8aee9d08): OFF ~14.56s (14.531-14.589) vs ON grid=4 ~14.49s
  (14.452-14.504). **Δ −0.07s (−0.5%), bands DON'T overlap** (ON max 14.504 < OFF min 14.531) = real,
  beats projected −0.04. Loss DIGIT-IDENTICAL 9.324 every step -> bit-exact gate PASS. (ppl 5th-digit
  wobble = run-to-run collective noise; OFF drifted +0.06 vs the 14.50 canonical -> WHY same-image A/B.)
- Applied to 14.50 canonical -> projects ~14.43 = new best. qk_diag_skip=True qk_diag_grid=4 is a KEEPER.
- FWD splash kernel has same diagonal waste = follow-up. NEXT: co-schedule A/B (image
  bwdagcosched-d5937c963), then compose qk_diag + co-schedule if both win. dkv-isolate xprof pending.

### qk_diag_skip mechanism CONFIRMED [2026-07-04, dkv isolate from qk_diag OFF/ON profiles]
- dkv kernel self-time: OFF 18.5ms/layer -> ON grid=4 17.4ms/layer = **−1.1ms/layer (−5.9%) MEASURED**.
  x58 MoE layers = −64ms/step ≈ the measured −0.07s step delta -> fully tied to the mechanism (skip
  the fully-masked diagonal QK sub-tiles). −5.9% vs handoff −3.8% = our H=128 vs their H=512 shape.
- qk_diag_skip=True qk_diag_grid=4 = KEEPER, fold into canonical (14.50 -> ~14.43 projected).

### qk_diag_skip FORWARD update [2026-07-04] — handoff v2: forward is the BIGGER win
- Updated handoff: same skip applies to the FORWARD splash kernel (splash_mha_fwd_residuals) = −14.8%
  vs prod (−8.4% vs best-retunable), BIGGER than the bwd −3.8%. Mechanism differs: fwd is softmax/VPU
  -bound; writing mask_value BEFORE the mask lets Mosaic constant-fold exp(mask_value−m)→0, DELETING
  the exp/softmax on the masked triangle (static exp-bundle −1344). Bit-exact (fwd O = 0).
- Our config MEETS the fwd gate (handoff confirmed against our exact prod shape: square 2048 blocks,
  MHA num_stacked_q_heads=1, HEAD_DIM_MINOR). Our manifest runs all sa_block_*=2048 (square).
- My c8aee9d08 port = spots 1+2 (config + bwd) only -> the measured −0.07s is the BACKWARD half.
  Resumed qk_diag agent to add SPOT 3 (forward, skip kj>qi) from branch pipelined-dkv-kernel @ ecb1d70.
  Then rebuild + re-A/B full fwd+bwd. Fwd should add more than the bwd's −0.07s.

### qk_diag FULL (fwd+bwd) staged [2026-07-04]
- Commit 4ee21b848 (branch qk-diag-skip, on c8aee9d08): forward spot added. Gates: flag-off
  byte-identical BOTH kernels; flag-on sub-grid compiles into fwd+bwd, FLOPs identical off-vs-on.
  Fwd gate met (num_stacked_q_heads=1 default, HEAD_DIM_MINOR default). Our sa_block_*=2048 square
  -> fwd skip active. exp-bundle -1344 fold = LLO metric (not local; measured in source 1f33c19/ecb1d70).
- Image qkdiagfull-4ee21b848 built+pushed. A/B PLAN: same-image off vs on, expect fwd+bwd combined
  win > the bwd-only −0.07s. This is the run to bank as the record (clean same-image absolute).
- Queued behind co-schedule A/B (coschedon still running).

### co-schedule A/B VERDICT [2026-07-04] — NO-GO: scheduling CYCLE (known gotcha)
- coschedoff (control) clean: 14.53s, loss 9.324 (validates bwdagcosched image = canonical).
- coschedon FAILED at compile: jax FAILED_PRECONDITION "A cycle is detected while visiting
  %reduce-scatter.115.cloned.1.call-done (transpose(jvp())/shard_map/reduce_scatter)". Co-tagging the
  backward w-ag + token-ag with the SAME _scheduling_group_id creates a DIRECTED CYCLE through the
  weight-grad RS dependency chain. Exact wall from [[wag-collective-coschedule]]: BACKWARD
  scheduling-group tags create cycles (forward-only tag avoids them). Fuse-serialize did NOT fire
  (agent proved diff replica groups) but the CYCLE gotcha did. Local CPU AOT couldn't catch it
  (TPU-scheduler-level; canonical AOT was blocked by busy dev TPU).
- Flag moe_bwd_ag_coschedule_group stays default-off; the idea needs cycle-avoiding surgical tagging
  (exclude the RS from the group / forward-informed) = R&D vs a known wall. DEPRIORITIZED vs qk_diag full.

### co-schedule cycle — advisor review dispatched [2026-07-04]
- Full directed cycle: reduce-scatter.115 (weight-grad RS) -> mpmd_map.142 -> pallas GMM/RS ->
  mpmd_map.148 (sort-bwd grad_hidden_states) -> sparse-core-data-format-call.41 -> reduce-scatter.115.
- Impl (d5937c963): token-AG tagged into group 1 (= w0 weight-AG's group). My read: token-AG and w-AG
  on OPPOSITE sides of the recompute->grad->RS chain -> the co-schedule ordering edge closes a loop
  against the data edges through the weight-grad RS. Classic backward-tag cycle (fwd-only avoids it).
- Advisor agent (Plan) reviewing: root-cause + fixable-vs-fundamental verdict + cycle-safe recommendation
  (own-group-id / optimization_barrier deadline / upstream-AG-instance / forward-only template). Parallel
  to the qk_diag full A/B.

### co-schedule ADVISOR VERDICT [2026-07-04] — FUNDAMENTAL wall, ABANDON scheduling-group
- Root cause: token-AG = activation INTO expert GEMM; reduce-scatter.115 = dW OUT of same GEMM ->
  OPPOSITE ends of the recompute->grad->RS chain (coexist in one HLO region because fused_bwd
  hand-assembles recompute+grad). Shared-group tag adds ordering edge RS.115.done->mpmd_map.142 (no
  data justification) -> closes cycle vs the real chain. RS.115 is TRANSITIVE (autodiff, untagged),
  pulled in by lying on the data path between the 2 tagged ops.
- ALL escapes REJECTED (advisor): own/new group id (singleton inert / fresh-shared recreates cycle);
  optimization_barrier (self-dual -> fences weight-grad RS exposed, moe.py:3186); upstream token-AG
  (none in bwd, it's the chain HEAD); forward-only template (bwd has no acyclicity guarantee).
  = exactly the [[wag-collective-coschedule]] wall.
- moe_bwd_ag_coschedule_group: documented NO-GO, default-off. ABANDON scheduling-group route.
- Redirect (placement not ordering, cycle-safe, UNMEASURED): (1) move bwd token-AG OFF SC queue -> ICI/TC
  (diff engine, can't serialize; HIGH cycle-safe / MED win); (2) hunt XLA multi-SC placement knob for
  dim-0/EP gathers (pure flag; HIGH cycle-safe / LOW-MED exists). Both DEPRIORITIZED vs qk_diag win.

### moe_direct_token_ag DISPATCHED [2026-07-04] — cycle-safe TC placement (advisor #1 + perf-drills)
- Path chosen after co-schedule cycle wall: get the BACKWARD token-AG OFF the SC offload queue onto TC
  (placement not ordering -> NO cycle). Mirror moe_direct_rs (_direct_reduce_scatter, moe.py:62-139,
  TC Pallas make_async_remote_copy, in canonical, STAGE2 ~83-86% auto-overlap). Build symmetric
  _direct_all_gather; flag moe_direct_token_ag, bwd-recompute-only (reuse the co-schedule plumbing shape).
- perf-drills receipts: weight_ag.py = TC-Pallas-AG ∥ SC-gather overlap mechanism proof; SCS_AG_PERF_NOTE
  = hand-rolled SCS ring 2.3x slower + fabric-contends -> TC path is the right answer (NOT SCS ring).
  CORRECTION to "TC AG beats XLA": XLA's lax.all_gather is fast on TC; the win is PLACEMENT/overlap.
- Gates: numerics == lax.all_gather; AOT flag-on token-AG is a TC pallas_call (not sparsecore) + COMPILES
  no-cycle; flag-off byte-identical. RISK (notes): on TC competes w/ GMM compute -> may re-expose;
  cluster A/B + xprof decides. Agent branch moe-direct-token-ag; parallel to qk_diag full A/B.

### qk_diag FULL (fwd+bwd) VERDICT [2026-07-04] — kernels win, STEP mostly doesn't (overlap-absorbed)
- Same-image A/B: OFF ~14.55 vs ON ~14.52 = −0.03s, BANDS OVERLAP (noisier than bwd-only −0.07).
  Loss digit-exact 9.324. Kernel isolate (measured): FWD splash 10.3->8.8ms/layer (−14.6%, matches
  handoff), BWD dkv 18.5->17.4ms/layer (−5.9%). Combined −2.6ms/layer x58 = −151ms IF 1:1.
- BUT step only −0.03..−0.07s -> the kernels are OVERLAPPED, not fully critical-path; savings absorbed
  by adjacent collectives. FWD splash MORE overlapped than dkv (sits under FSDP AGs) -> the "fwd is the
  bigger win" did NOT translate end-to-end. CORRECTION: my earlier 14.43 projection was WRONG (assumed
  kernel Δ flows 1:1; it doesn't).
- VERDICT: qk_diag_skip = bit-exact but MODEST ~−0.03..−0.07s step (noise-comparable ±0.05). KEEP (free,
  bit-exact, directional+), fold into canonical, but NOT a banked record without noise-averaged re-measure.
- STRATEGIC: compute-kernel wins are overlap-absorbed -> the real lever is EXPOSED COLLECTIVE time, not
  kernel FLOPs. = exactly what moe_direct_ag attacks (token-AG off the serialized SC queue). Watch that one.

### moe_direct_token_ag LANDED + A/B LAUNCHED [2026-07-04]
- Commit 210f818ae (branch moe-direct-token-ag), image directag-210f818ae. _direct_all_gather =
  symmetric mirror of _direct_reduce_scatter (TC Pallas make_async_remote_copy; custom_vjp bwd =
  psum_scatter, no SC Pallas -> no cycle). Bwd-recompute-only, big x gather.
- AOT receipts (measured): flag-off byte-identical (0 HLO diff); flag-on COMPILES no-cycle (EXIT 0 --
  the whole point vs co-schedule); backward token-AG MOVED SC->TC (SC device-ops 82->81, TC
  custom_call 22->23, sparsecore-AG 176->175 = the 1 recompute token-AG now a TC pallas_call, NOT
  sparsecore-offloaded); peak mem unchanged. CPU bit-exact vs lax.all_gather all shapes. Option-2
  (offload exemption) = no clean per-op lever (global flags only) -> kernel is the route.
- A/B: ds-v3-dtagoff/on-0704 (same image, moe_direct_token_ag off vs on) on plain canonical (no qk_diag
  noise). Acceptance: loss digit-exact; step delta; xprof = does the TC token-AG overlap the SC w-AG.
  RISK: on TC competes w/ recompute GMMs -> may re-expose (A/B + xprof decides).

### moe_direct_token_ag VERDICT [2026-07-04] — NEW RECORD 14.33s, CLEAN −0.20s
- Same-image A/B (image directag-210f818ae): OFF ~14.53 (14.516-14.558) vs ON ~14.33 (14.322-14.347).
  **Δ −0.20s (−1.4%), BANDS DON'T TOUCH** (OFF min 14.516 > ON max 14.347) = cleanest of the day, NOT
  noise. Loss digit-exact 9.324 (== lax.all_gather). **NEW RECORD 14.33s** -- decisively beats the
  all-time 14.54 (old branch) AND the 14.50 canonical.
- MECHANISM (profile, backward lane abs ms): exposed collective 2.76->2.31 (−0.45s, dispatch token-AG
  off the serialized SC queue), compute +0.37 (AG now a TC pallas_call, partly the re-expose risk but
  mostly overlapped), net bwd −0.20 = the whole step win. Confirms: EXPOSED COLLECTIVE is the lever,
  not kernel FLOPs (qk_diag lesson). TC-placement (moe_direct_rs trick) applied to the AG WINS.
- CANONICAL UPDATED: canonical_best_14.33_manifest.yaml (moe_direct_token_ag=True). Profiles in
  dropdown: directAG-OFF-14.53 / directAG-ON-14.33-BEST.
- CLARIFICATION: this moved the DISPATCH token-AG (route recompute, bf16[4,4096,7168]), NOT .626 (the
  combine cotangent, bf16[32768,7168], the 9.4ms exposed one). .626 is a SEPARATE, BIGGER target ->
  next: a direct-AG for the combine cotangent (transpose of moe_direct_rs, also SC-offloaded).

### Two experiments dispatched on top of 14.33 [2026-07-04]
- (1) CHEAP PROBE (user): AOT-only check — does co-tagging .626 (combine cotangent AG) + .527 (w-ag)
  cycle like the dispatch-AG co-schedule did? Prior: advisor says bwd co-schedule is a fundamental
  cycle; "separate SCs" doesn't dissolve it (cycle = dataflow, not placement); .626 is MORE downstream
  -> expect cycle, but .626 is a diff op so verify. Agent probe-626-coschedule, AOT verdict only.
- (2) MAIN BET (user): .626 combine-cotangent direct-AG (TC kernel, reuse _direct_all_gather; transpose
  of moe_direct_rs). BIGGER exposed target (9.4ms, 3 serial single-SC transfers) than the dispatch AG
  that just won −0.20s. Agent moe-direct-combine-ag. Caveat: .626 dep-locked + already partial-overlaps
  GMM -> less certain than dispatch AG; cluster A/B decides. Both parallel, on top of the 14.33 record.

### Two .626 levers ready [2026-07-04]
- (1) co-schedule .626+wag (probe a71296208, image cscombwag-a71296208): CYCLE-FREE (surprise -
  advisor's cycle was dispatch-AG-specific; .626 sits differently). A/B RUNNING (cscwoff/on). Cheap
  (flag only, no kernel). Tests if the scheduling group actually overlaps .626 with the w-ag.
- (2) .626 direct-AG (moe_direct_combine_ag, commit 1bae5dc9c, image directcombag-1bae5dc9c): TC Pallas
  _direct_all_gather on the combine cotangent (transpose of moe_direct_rs). Gates: bit-exact; flag-off
  byte-identical; flag-on compiles no-cycle, combine-bwd AG moved SC->TC (chunked-combine-rs-bwd AG
  1->0, tpu_custom_call 29->30); peak -64KB. STAGED, A/B after the co-schedule.
- Both attack the SAME .626 (9.4ms exposed, bigger than the dispatch AG's −0.20s win). Compare head to
  head; keep the winner. Then final compose: direct_token_ag + qk_diag + winning .626 lever.

### co-schedule .626+wag VERDICT [2026-07-04] — REGRESSION +0.20s, NO-GO
- Same-image A/B (cscombwag): OFF ~14.57 (14.532-14.620) vs ON ~14.77 (14.739-14.815) = **+0.20s
  regression, bands don't overlap**, loss digit-exact. Compiled cycle-free BUT the scheduling-group
  tag HURT: co-locating .626 with the w-ag broke .626's existing partial overlap with the preceding
  GMM -> MORE exposed. The fuse/serialize gotcha (memory: a tag can serialize two collectives).
- LESSON crystallized: in this backward, PLACEMENT (direct-AG, move to TC) helps; SCHEDULING-ORDERING
  (group tag) hurts (cycle OR serialization). Scheduling-group route CLOSED for the backward.
  moe_coschedule_combine_wag default-off, NO-GO.
- Remaining .626 lever: the direct-AG (directcombag-1bae5dc9c) -> A/B now.

### .626 direct-AG VERDICT [2026-07-04] — WIN −0.10s clean (2nd placement win)
- Same-image A/B (directcombag-1bae5dc9c, 14.50 base): OFF ~14.54 (14.519-14.559) vs ON ~14.43
  (14.427-14.442) = **−0.10s, bands don't overlap**, loss digit-exact. Smaller than dispatch AG's
  −0.20 (as predicted: .626 already partial-overlaps the GMM -> lower ceiling), but CLEAN real win.
- .626 scoreboard DECISIVE: co-schedule REGRESSED +0.20 | direct-AG WON −0.10. PLACEMENT >> ORDERING.
- Two independent placement wins: moe_direct_token_ag (−0.20 -> 14.33) + moe_direct_combine_ag (−0.10).
  Both move an SC-offloaded EP AG to TC. FINAL COMPOSE dispatched: direct_token_ag + direct_combine_ag
  + qk_diag_skip merged -> one image -> A/B for the definitive best (projection ~14.2, compose measures
  the true stacked number since both AGs add TC DMA traffic).

### FINAL COMPOSE A/B LAUNCHED [2026-07-04]
- Branch compose-all-levers (tip 5c01ac873), image composeall-5c01ac873 (Dockerfile.qkdiag = tokamax
  overlay + merged src). Cherry-pick stack: moe_direct_token_ag(id40) + moe_direct_combine_ag(id50) +
  qk_diag_skip. Shared _direct_all_gather -> 1 def, 2 collective-ids. Flag-off gated byte-identical
  (derived; all-on AOT not re-measured, TPU busy - inherits per-branch gates).
- A/B: ds-v3-composeoff (all 4 flags OFF, control -> should = 14.50, validates merged image) vs
  ds-v3-composeon (moe_direct_token_ag + moe_direct_combine_ag + qk_diag_skip + qk_diag_grid=4).
  Early cycle/OOM watch (2 direct-AGs may interact). Projection ~14.2 IF the -0.20 + -0.10 + -0.05
  stack; compose measures TRUE number (both AGs add TC DMA traffic -> may not be fully additive).
  Acceptance: loss digit-exact 9.324.

### FINAL COMPOSE VERDICT [2026-07-04] — NEW RECORD 14.10s, SUPER-ADDITIVE
- Same-image A/B (composeall-5c01ac873): OFF ~14.53 (14.501-14.563) vs ON ~14.10 (14.088-14.135) =
  **Δ −0.43s, bands don't overlap**, loss digit-exact 9.324, compiled cycle-free (2 direct-AGs coexist).
- **NEW RECORD 14.10s.** Levers: moe_direct_token_ag(−0.20) + moe_direct_combine_ag(−0.10) +
  qk_diag_skip(−0.05). Naive sum −0.35 but MEASURED −0.43 = SUPER-ADDITIVE (freeing BOTH EP AGs off
  the serialized SC queue helps more than each alone; qk_diag TC headroom aids the overlap).
- CANONICAL = canonical_best_14.10_manifest.yaml (branch compose-all-levers 5c01ac873, image
  composeall-5c01ac873, 4 flags: moe_direct_token_ag + moe_direct_combine_ag + qk_diag_skip +
  qk_diag_grid=4). Profile: compose-ALL-14.10-RECORD in dropdown.
- DAY ARC: 15.33 -> 14.10 = **−1.23s (−8.0%)**, −0.44s under the old all-time 14.54. Throughline:
  EXPOSED-COLLECTIVE PLACEMENT (move SC-offloaded EP AGs to TC) is the lever; compute-FLOP wins are
  overlap-absorbed; scheduling-ORDERING (co-schedule groups) hits walls (cycle/serialize).

### moe_bwd_inkernel_quant VERDICT [2026-08-14] — rbf-dependent lever, retires the rbf=-1 NaN hazard
- **Motivation (from the rbf profile A/B):** `compare_profiles` of the record (siv-cn-xsretest, rbf=2,
  4.251s) vs siv-cn-rbfoff2 (rbf=-1, 5.474s) attributed the +1.22s to **VPU +962ms / MXU only +194ms**.
  The ragged kernels skip empty groups fine (m_32768 -> m_131072 cost the MXU almost nothing); the tax
  was every *dense* elementwise op sized by the buffer's STATIC shape: fp8 clamp_convert on
  [131072,7168] (~565ms), the sanitizer's bf16 select (~400ms, 1.7ms x232 vs 2us at rbf=2), abs_reduce
  amax over bf16[131072] (~160ms).
- **Change (commit dafb04634):** quantize the backward operands INSIDE the ragged kernels.
  dlhs: skip the XLA cotangent quantize; gmm_v2's quantized-matmul path now also fires for a wide lhs +
  UNSCALED fp8 rhs (per-row per-512-block e4m3 in VMEM). drhs: tgmm_v2 `quantize_operands` quantizes
  BOTH operands per-gm-tile-per-channel with the scale outer-product applied per tile (the tgmm_block
  algebra fused into the tile loop) -- subsumes 3 dense ops (x_sorted re-quantize, drhs_dout*=lhs.scale,
  per-N cotangent quantize).
- **MEASURED (8x8x8 pdbs=1 synthetic, image 1410-up2-dafb04634, all loss 8.784 at step 19):**

  | arm | rbf | quant | sanitizer | s/step | TPS/chip |
  |---|---|---|---|---|---|
  | siv-cn-xsretest (RECORD) | 2 | dense | off | 4.251 | 1927 |
  | siv-cn-ikqon | 2 | in-kernel | off | 4.303 | 1904 |
  | siv-cn-rbfoff2 | -1 | dense | **on (required)** | 5.474 | 1497 |
  | siv-cn-ikqrbf3 | -1 | in-kernel | **OFF** | 4.882 | 1678 |

- **VERDICT: the lever's value is buffer-size-dependent.** At rbf=2 it is +0.05s (net-negative, inside
  noise but not a win): the dense quantize family was only ~250ms there, and the tgmm now reads bf16
  operands (2x the bytes of the pre-quantized e4m3 it used to read), which eats the savings. At rbf=-1
  it is **-0.59s (+12% TPS)**, recovering ~48% of the 1.22s rbf penalty.
- **SECOND RESULT (arguably the bigger one): the rbf=-1 stale-row NaN hazard is GONE.** ikqrbf3 ran
  clean for 20 steps with `moe_sanitize_ragged_buffer` OFF, where the same stack with dense quant
  NaN'd at step 1 (siv-cn-rbfoff). Mechanism: the NaN door was the dense per-row amax ingesting
  uninitialized HBM rows (amax=Inf -> inv=0 -> Inf*0=NaN). Every quantize is now group_sizes-bounded,
  so no reduce ever reads the buffer tail. The sanitizer flag becomes unnecessary at rbf=-1 rather
  than merely masking the issue.
- **Loss bit-identical 8.784 across all four arms** -- finer per-tile e4m3 scales are numerically
  neutral at this horizon (20 steps, synthetic). Real-data curve check still owed before any default flip.
- **Residual gap rbf=-1 vs rbf=2 is now 0.63s** (was 1.22s). The remainder is the 4x-larger
  reduce-scatters (RS.31 987->1126ms, RS.35 466->596, RS.33 392->588; mostly hidden, SC lane +48ms
  exposed), relayout +132ms, and the tgmm's bf16 operand reads over 4x rows.
- **DON'T flip on at rbf=2.** Keep the record stack as-is; this flag is for rbf=-1 configs (real-data
  runs where rbf=2 truncation-drop under imbalanced routing is an open accuracy question, and where
  rbf=2 currently hits the gmm_v2 init-fatal on real routing).
- Infra note: 2 launches lost to flakes before ikqrbf3 -- one jobset never composed its slice (Warden
  `tpu-accelerator-topology-constraints` while ss-kueue-operator sat at `1 CREATED`), one hit the
  gang-formation init hang ending in the `TearDownMesh` HAL abort (dies in make_tpu_client, pre-compile).
  Both cured by delete+relaunch. Babysitters now carry a stalled-init detector (18min, 0 steps -> bail).

### CORRECTION to the entry above [2026-08-14] — the dlhs half is NEEDED; my profile inference was wrong
Two follow-up arms overturned the mechanism I wrote above. Recording both the correction and the
methodological miss, because the miss is the reusable part.

- **What I claimed:** from op-family accounting in the ikqrbf3 profile I concluded the dlhs half was
  net-negative by ~678ms. The observation was real: `_dlhs_scale_grad_by_rhs_scale`'s multiply had been
  FUSED INTO the XLA quantize (fused form writes fp8, 1 B/elem); dropping the quantize makes it
  materialize bf16 (2 B/elem) and the kernel re-reads 2x. Op pair went 1128ms -> 1806ms, MEASURED.
- **What the controlled A/B says (siv-cn-ikqd1, drhs-half only, rbf=-1): 5.129s** -- WORSE than both
  halves (4.882) and barely better than dense (5.474). **The dlhs half is worth -0.247s, not +0.678s.**
- **The miss:** I compared op-family self-times ACROSS two arms and read a causal delta out of it. Self
  time is not step time -- the +678ms of un-fused bf16 multiply is largely absorbed by overlap, while
  what the dlhs quantize removal actually buys (its own amax + a serialized VPU pass on the critical
  path) is not visible in a self-time table. **Op accounting proposes; only a same-image A/B with the
  single flag flipped disposes.** This is the CLAUDE.md "a measured win validates the change, not the
  mechanism" rule, hit from the other direction: a measured op delta did not validate a mechanism.
- **acc_dtype confound, resolved.** gmm_v2 defaults acc_dtype to bf16 whenever it quantizes the lhs
  in-kernel, so ikqrbf3's 4.882 ran the dlhs GRADIENT accumulator in bf16. Pinned to f32
  (commit a5415ab9f) and re-ran: **siv-cn-ikqd2 = 4.884s** -- identical within noise. The win is NOT
  an accumulator downgrade; keep the f32 pin (same speed, safe numerics). Caught by advisor review,
  not by any gate I had -- 20 synthetic steps cannot see an accumulator change.

**FINAL rbf=-1 ladder (all loss 8.783-8.784):**

| arm | config | s/step | TPS/chip |
|---|---|---|---|
| siv-cn-rbfoff2 | dense quant + sanitizer (required) | 5.474 | 1497 |
| siv-cn-ikqd1 | in-kernel drhs only | 5.129 | 1597 |
| siv-cn-ikqrbf3 | in-kernel both, bf16 acc | 4.882 | 1678 |
| **siv-cn-ikqd2** | **in-kernel both, f32 acc pin (KEEP)** | **4.884** | **1677** |

- **Ship state:** `moe_bwd_inkernel_quant=true moe_bwd_inkernel_quant_dlhs=true` at rbf=-1 = 4.884s,
  **-0.59s / +12% TPS vs dense, sanitizer not needed**. At rbf=2 still skip (+0.05s). Residual gap to
  the rbf=2 record is 0.63s: TC-bound at 3.94s ceiling, VPU now 1.65s (was 2.18s), relayout 138ms,
  SC 936ms exposed.
- The two flags are now effectively one; keep them separate only until a real-data curve confirms.
