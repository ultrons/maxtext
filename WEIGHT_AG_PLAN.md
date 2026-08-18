# Plan: hide the weight all-gathers on the 6.8 s baseline

Target: `siv-cn-rwag0`, 6.864 s/step, real c4 tokens, real gate, MTP + grouped routing, fp8 ship
stack, rbf=-1, chunks=2, 8x8x8 / fsdp=128 / EP=8.

## 1. What we are hiding

Exposed all-gather is **1274 ms/step, 18.8% of the step** (5278.95 ms over 4.14 profiled steps).
That is the ceiling on this plan; nothing here can beat it.

Complete census from the optimized HLO (81 all-gather ops; `WEIGHT(spmd)` = SPMD-inserted on the
quantized kernel). Dims verified against `deepseek3-671b.yml`, not recalled.

| weight | shape | fwd | bwd-remat |
|---|---|---|---|
| MLA q_a | `[1,7168,1536]` | 2 | 4 |
| MLA kv_a | `[1,7168,576]` | 2 | 4 |
| MLA q_b | `[1,1536,128,192]` | 2 | 2 |
| MLA kv_b | `[1,512,128,256]` | 2 | 2 |
| MLA o_proj | `[1,128,128,7168]` | 2 | 2 |
| dense MLP wi / wo | `[1,7168,18432]` / `[1,18432,7168]` | 2 / 1 | 2 / 1 |
| shared expert wi / wo | `[1,7168,2048]` / `[1,2048,7168]` | 2 / 1 | 2 / 1 |
| router gate | `[1,7168,256]` | 1 | 1 |

Two facts drive everything below. **Every weight is a fwd + bwd-remat pair**, so a forward-side fix
that survives remat covers both members. And the cost is concentrated: `.445` (bwd-remat, 0.3 GB/s,
~536 ms/step) plus `.525` (fwd, 1.1 GB/s, ~419 ms/step) are **75% of all exposed gather time** in
two of the 81 ops. This is not a broad sweep, it is two ops plus a guard against regressing the rest.

The routed-expert gathers are already ours (`shard_map`) and run at **80.8 GB/s on 245 MB**. Same
fabric, same replica groups, 250x the bandwidth of `.445`. Placement is the defect, not the fabric
and not XLA's collective implementation. Keep the kernels thin accordingly: start/done wrappers
around the same DMAs, not a rewritten collective.

## 2. The mechanism, measured

`probes/startdone_runtime.py`, v7x 2x2x1, jax 0.10.1. `start` arms the DMA and returns; `done`
RECONSTRUCTS the identical descriptor and waits. No semaphore is passed; each kernel allocates its
own as scratch, which is what puts it in sync-flag memory.

| gate | result |
|---|---|
| R1 reconstructed descriptor waits on the armed DMA | PASS, `0.000e+00` |
| R2 ordering holds with compute between the halves | PASS, `0.000e+00` |
| R3 remote cross-device | PASS, `0.000e+00` |
| R4 matmul next to an in-flight DMA | PASS, 819.4 -> 789.2 TFLOP/s (96.3%) |
| R5 MISMATCHED scratch | **HANGS** (compiles, then never returns) |
| R6 MATCHED scratch on both halves | PASS, `0.000e+00` |

**Hard rule from R5/R6: both halves must declare byte-identical `scratch_shapes`**, even where one
half never uses the padding. Mosaic assigns the DMA semaphore a different slot under a different
footprint, so the wait targets the wrong sync flag. It fails as a hang, never as an error. Every
kernel pair we write carries a shared scratch signature, and a unit test asserts the two lists are
equal so this cannot regress silently.

R4 also bounds the `mpmd-map-sc-barrier-blocks-fsdp-ag` worry (SC kernel next to a gmm poisoned an
AG 47 -> 3 GB/s): a TC-shaped custom call does not do that. Bounded, not eliminated -- clean
microbenchmark versus a contended model.

## 3. The hiding budget says within-layer placement is enough

| | per layer (58) | per layer (61) |
|---|---|---|
| TC compute available | 45.2 ms | 43.0 ms |
| exposed all-gather | 22.0 ms | 20.9 ms |
| **ratio** | **2.06** | **2.06** |

There is roughly twice the compute needed to cover the gathers *within* a layer. So `start` at the
top of the layer and `done` immediately before the consuming matmul should suffice, and we do NOT
need cross-layer prefetch or the scan-carry plumbing `swap_gather_w01` required. That is the single
biggest simplification available. If the measured margin turns out thinner, cross-layer returns as
the fallback.

## 4. Build order

**Step 0 (blocking, cheap): does start/done survive remat?**
AOT the forward-only start/done and count pairs in both scopes. If the dump shows the pair under
`closed_call` AND under `rematted_computation`, one forward-side change covers both members of every
census pair and the plan is a handful of edits. If remat collapses or reorders the pair, the backward
needs the hand-written bwd path and the plan roughly doubles. Watch for the `FAILED_PRECONDITION`
scheduling cycle that annotations triggered; start/done is structure rather than metadata so it may
not fire, but that is the failure to look for.

**Step 1: pin `.445`.** Still unidentified. It is a bwd-remat weight gather at 5.0-5.3 MB, which
matches NO census shape cleanly (kv_a is 4.13 MB, q_a is 11.0 MB), and the same instrument reports
`.525` as 15.3/16.0 MB where the true shared-expert weight is 14.68 MB, so the tool's byte figure
runs high by roughly 9% for reasons I have not established. `show_hlo --optimized` does not work
here: the xplane carries no HLO. Get it by re-running with `--xla_dump_to` so the run's own HLO
matches the profile's op numbering, or re-profile with `--xla_enable_custom_call_region_trace=true`.
Do not target `.445` on a guess between q_a and kv_a.

**Step 2: one weight end-to-end.** The forward shared-expert gather first -- not the biggest, but the
plumbing is proven there and there is a measured A/B baseline, so a delta is attributable.
Gates per conversion: AOT compile -> 2x2x1 runtime correctness **with the real weight shape** (not
the 8192-element toy) -> cluster A/B.

**Step 3: generalize** to the MLA projections, dense MLP, gate, in descending measured cost.

## 5. Attribution discipline

Accept a conversion only when the step time drops **and** the exposed stall for that op drops **and**
the op's source moves to our kernel. A step win without the stall moving means something else moved
and we would bank a result we cannot reproduce.

Run the start/done A/B against the **stock 6.864 s baseline with the hoist off**. The shipped hoist
is default-off pending an unexplained 0.174 lm_loss delta (above the ~0.09 noise floor); stacking on
top of it would give two unexplained numerics sources and no way to attribute either.

Substrate: real c4 tokens with `reuse_example_batch=1`. A synthetic run is 4.965 s and leaves
SparseCore near-idle, which removes the contention this work exists to fix.

---

# Design as built (2026-08-18)

## The problem in one line

The FSDP weight all-gathers are exposed for **1274 ms/step, 18.8% of a 6.864 s step**, and they are
slow because of WHERE XLA puts them, not because of the fabric or the collective implementation.
The evidence: `all-gather.519` moves 245 MB at **80.8 GB/s** while `all-gather.445` moves 5.3 MB at
**0.3 GB/s**. Same op, same replica groups, 46x the bytes, 250x the bandwidth. Two of 81 ops carry
75% of the exposed time.

## Why we stopped asking XLA nicely

Four attempts to influence the scheduler all failed, and each failed for a different reason worth
remembering. A region tag on the layer hits `annotation groups with gaps`. A narrow tag on the
kernel materialization lands on a cast XLA folds away (2 ops carried the id, neither a gather). A
`with_sharding_constraint` hoist gets ABSORBED by SPMD, which re-emits its own all-gather and drops
our frontend attribute entirely. Only an explicitly emitted collective (`shard_map` + `lax.all_gather`)
kept the tag, and that won -0.208 s -- but its placement is still XLA's to choose.

So: own the transfer outright.

## The mechanism: split-phase start/done

Two ordinary TC-shaped Pallas custom calls. `start` arms the DMAs and returns WITHOUT waiting;
`done` RECONSTRUCTS the identical descriptors and waits. Because both look like normal compute ops,
XLA schedules around them instead of fencing at them, and whatever we place between them overlaps
the transfer. Ordering is enforced by a real buffer dependency (`done` takes `start`'s output), since
the semaphore is invisible to the scheduler.

No semaphore is passed between the kernels. Each allocates its own as scratch, which is what puts it
in sync-flag memory; Mosaic's deterministic allocation makes `done`'s rebuilt descriptor name the
flag `start` armed.

Measured on v7x: R1 delivery exact, R2 ordering holds with compute in the gap, R3 remote exact,
R4 a neighbouring matmul retains **96.3%** of throughput (which retires the old `mpmd_map` worry that
a co-resident kernel poisons collectives), R6 the scratch rule. R5 is the trap: **mismatched
`scratch_shapes` between the halves HANGS** rather than raising, so both halves share one signature
and a guard asserts it.

## The kernel: static destinations, no indexing

The natural push formulation (`o_ref.at[me]`) is REFUTED on hardware: the offset is computed on the
sender and must be interpreted in the receiver's buffer, and the bytes silently never arrive at all.
A pull model does not fix it -- the offset still crosses the boundary.

What works: never index. At step `k` (`k = 1..n-1`, a PYTHON constant) device `me` pushes its shard
into device `(me+k)`'s **buffer k**. `k` is static on both sides, each step has its own output buffer
and its own send/recv semaphore pair, and the gathered array is assembled outside the kernel with
ordinary XLA ops. Verified delivering every shard on 8 devices.

## The backward

`lax.psum_scatter`, i.e. XLA's reduce-scatter, deliberately. The reduce-scatters measure 10.4-25.9
GB/s against the gathers' 0.3-1.1, and `rs-lever-mapped-closed` already found direct-to-owner RS
near-optimal. The RS is not the pathology; writing an accumulating kernel would be effort on the
healthy collective. Crucially the big backward item, `.445`, is an all-GATHER inside
`rematted_computation` -- the forward gather re-run -- so it is covered by the forward conversion.

## Where it plugs in, and why one change covers two gathers

Inside `DenseGeneral`, ahead of the dot, replacing the SPMD-inserted gather. Every weight in the
census appears as a **fwd + bwd-remat pair**, and Step 0 proved the start/done pair RE-TRACES into
`rematted_computation` intact (6 pairs in each scope, no scheduling cycle). So one forward-side
conversion covers both members.

Placement is WITHIN a layer: `start` at the top, `done` before the consuming matmul. The budget says
that is enough -- 45.2 ms/layer of TC compute against 22.0 ms/layer of exposed gather, a ratio of
**2.06** -- so no cross-layer prefetch and no scan-carry plumbing.

## Non-negotiable requirements (each cost an iteration to learn)

1. Byte-identical `scratch_shapes` on both halves, or it hangs.
2. `custom_vjp` wrapper: a `pallas_call` on a grad-live path cannot be differentiated.
3. `shard_map` wrapper: Mosaic kernels cannot be auto-partitioned.
4. `input_output_aliases` is mandatory, not optional: without it `done` returns a buffer nothing wrote.
5. DMA source and destination shapes must match exactly.

## What is left

Scale (127 buffers / 254 semaphores at fsdp=128 is untested and could invalidate the width-based
design; fallback is a ring with a static double buffer), then swap into `kernels/startdone.py` and
re-gate numerics + VJP bit-exactly, AOT, then a cluster A/B against the **stock 6.864 s baseline with
the hoist off** (the hoist is default-off pending an unexplained 0.174 lm_loss delta, and stacking
would leave two numerics sources).
