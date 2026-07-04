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
