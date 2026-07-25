# Port plan: SC-fused combine+reduce-scatter → upstream-head ladder (rung6-sliced)

Scoping assessment, 2026-07-03. NO code written. Sources: `fused-combine-rs` branch in
`~/maxtext` (commits `b5bfd57b2..e03ab077a`), `~/perf-drills/gather/combine/` (playbook +
prototypes + result notes), this tree (`rung6-sliced`: `chunked_ring_combine_reduce_scatter`
+ per-chunk barriers + manbwd).

Provenance tags: **[measured]** = traced to a command output / receipt file in a prior
session's notes cited by path; **[derived]** = arithmetic from measured inputs;
**[projected]** = expectation, needs measurement. The 17.3 ms = 9.5 combine + 7.8 SC-offloaded
RS tail decomposition is from this session's profile forensics (parent context) and is
treated as measured.

---

## 1. Inventory

### 1a. What exists on the old `fused-combine-rs` branch (~1050 LOC over 8 files)

| File | Content |
|---|---|
| `src/maxtext/kernels/fused_combine_rs.py` (402 LOC) | **The production-hardened kernel pair.** `fused_combine_rs_kernel` (fwd) + `fused_combine_rs_bwd_kernel` (bwd, unused in final wiring). This is the A3 iteration: full-mesh multi-axis `device_id`, `held_idx` SMEM input (scan-nesting constvar fix), H-sub-tiling (`HC=512`), token sub-tiling (`token_tile=16`, added after the production VMEM blocker). **Port THIS file**, not `fcr_kernel.py`. |
| `src/maxtext/kernels/fcr_kernel.py` (209 LOC) | Earlier importable version (in-kernel `lax.axis_index` device_ids — breaks under `nn.scan`; no token tiling in bwd). Reference only; do not port. |
| `src/maxtext/layers/moe.py` (+261) | `_fused_cr_inputs` (routing→kernel adapter), `_fused_combine_rs` (composed `custom_vjp` wrapper: pure-JAX `linear_transpose` bwd, `sidx` threaded as arg to defeat tracer escape), `_direct_reduce_scatter` (separate TC-Pallas RS lever, NOT part of this port), branch in the `use_ring_of_experts` output path replacing `unpermute`+`psum_scatter`. |
| `src/maxtext/layers/decoders.py` (+14) | Remat-policy plumbing: `checkpoint_name("moe_fused_combine_rs")` folded into every remat policy so the layer remat SAVES (does not re-run) the SC kernel in backward. |
| `src/maxtext/configs/{types.py,base.yml}` (+25) | Flags `moe_fused_combine_rs`, `moe_direct_rs`. |
| `flags_A.txt`, `repro_variant.sh` | Production XLA flag set (incl. `xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true`) + xpk repro. |
| Tests | **None in-repo.** Validation lived in `~/perf-drills/gather/combine/` (v5p equivalence + `v7x_maxtext_validate.py` harness) and cluster flag-flips. |

### 1b. Kernel contract (the exact thing being ported)

`fused_combine_rs_kernel(buf, sidx, held_idx, ep_name, mesh_axes, mesh_shape, *, n_tokens, hidden, k, token_tile=16, scatter=True) -> [CH, hidden] f32`, `CH = n_tokens // EP`.

- **Inputs**: `buf [BUF, H] f32` — expert-output buffer with routing weights **pre-folded per
  row** (`buf = interm * w_buf[:, None]`, caller pre-casts to f32); `sidx [n_tokens*k] int32` —
  token-grouped gather indices (`out[t] = Σ_k buf[sidx[t*k+kk]]`), owner-block-ordered so slots
  `[c·CH·k, (c+1)·CH·k)` belong to RS-owner `c`; `held_idx [n_held] int32` — the runtime
  `axis_index` of every size>1 non-EP mesh axis, computed OUTSIDE the kernel (the scan-nesting
  "No constant handler" fix — in-kernel `axis_index` inside a nested `run_scoped` device_id
  constvars).
- **Structure**: `pl.kernel(body=[go_scs, go_tec], mesh=[ScalarSubcoreMesh(num_cores=1),
  VectorSubcoreMesh(num_cores=1, num_subcores=1)])`. TEC: per EP-owner-chunk, token-tiled
  (TT=16) × H-tiled (HC=512) per-row `make_async_copy` gather (NL rows/tile, NL from
  `get_tpu_info().sparse_core.num_lanes`) → K-reduce → `partial_ref` (double-buffered HBM);
  after `scatter_done`, EP-sum of `recv_ref`. SCS: `ready` handshake, then per chunk
  `async_remote_copy(partial[c%2] → peer-c recv[my])` with dict `device_id` (EP axis varies,
  all other axes held at `held_idx`/0), streamed one chunk behind TEC.
- **Semaphores**: `b_scs/b_tec` (init barrier, SCS signals both), `tec_to_scs`/`scs_to_tec`
  (chunk handshake), `scatter_done`, `ready` — all REGULAR; DMA sems via `run_scoped`.
- **Scratch**: HBM `partial (2,CH,H) f32` + `recv (EP,CH,H) f32`.
- **Dtype**: f32 ONLY (SC vector `Get` bf16 supports shapes `(32,)`/`(2,16)` only; the NL-wide
  lane reads in the reduce are f32).
- **Static asserts**: `(TT·k) % NL == 0` (v7x: 16·8=128 % 16 ✓), `CH % TT == 0`,
  `H % HC == 0`, `HC % 128 == 0` (H=7168/HC=512 ✓).
- **Ablation knob**: `scatter=False` runs TEC-only (combine without RS) — the built-in A/B for
  "is the RS hidden".

**Backward**: `fused_combine_rs_bwd_kernel` = exact transpose (SCS all-gather `d_out`, TEC
scatter-write). It EXISTS and was v5p-verified rel=0, but the **final branch wiring does NOT
use it**: an SC Pallas kernel in a `custom_vjp` bwd rule constvars under `lax.scan`
(memory: sc-kernel-custom-vjp-scan-constvar). Shipped bwd = pure-JAX
`jax.linear_transpose` of `take → K-sum → psum_scatter` (commit `a1245eba9`), with `sidx`
passed as a custom_vjp argument (commit `e03ab077a`, tracer-escape fix).

### 1c. Validation status of the old integration (be honest about what ran)

- v5p EP=4: fwd rel=0 vs `gather+reduce+psum_scatter`; RS add ≈0.5% wall (0.907 vs 0.902 ms)
  [measured, `HANDOFF_archA_fused.md`].
- v7x 4×8×8 harness (`v7x_maxtext_validate.py`) at **NTOK=64 toy shape**: rel=0.005 — the
  cross-chip SCS `async_remote_copy` + full-mesh device_id DID work on real v7x
  [measured, `PROD_NTOK_VMEM_BLOCKER.md`].
- v7x production flag-flip (deepseek3-671b, 4×8×8, prod batch): **failed at compile** —
  `CompileTimeSparseCoreAllocationFailure`, `vg = CH·K×HC = 131072×512` words, 512× over the
  131071-word cap (CH=16384, K=8) [measured, same note]. The token-tiling fix (TT=16) was
  committed AFTER this (`1f5e3ea21`), and **no successful production-shape v7x run of the
  token-tiled kernel is on record**. "Complete, gated on kernel A3-bwd + v7x" (memory) means:
  wiring done, kernel never proven at production shape on v7x.

### 1d. What rung6-sliced's chunked path needs (the target contract)

`chunked_ring_combine_reduce_scatter` (`src/maxtext/kernels/ragged/ragged_sort.py:482`):
- Per chunk `c`: slice `ridx_c, w_c` (O(T/N) sliced operands, tpu-inference style; the
  expert-sorted buffer `sorted_tokens_local` is read WHOLE), `optimization_barrier` fence on
  the previous chunk's PRE-RS output (un-fuse the cross-chunk converts),
  `ring_ragged_unsort(...)` → `psum_scatter` → concat. Token axis pre-permuted chunk-major
  (`_permute_tokens_for_chunked_rs`) so per-chunk RS + concat reassembles global order.
- `ring_ragged_unsort` handles TWO buffering modes (global-position vs truncated/packed
  buffer, decided by `full_num_slots`), ragged `group_sizes`, validity masking (rows outside
  this shard's `[start,end)` → 0), bf16 in/out (f32 accumulate inside), and carries the proven
  hand-written combine bwd.
- The combine kernel it calls (`ragged_gather_reduce`) runs on the **full SparseCore**
  (`VectorSubcoreMesh(num_cores=sc_info.num_cores, num_subcores=sc_info.num_subcores)` —
   v7x: 2 cores × 16 subcores) with row/column partitioning across subcores.
- Backward: rung 6 default (`moe_chunked_combine_in_remat=False`) — the manbwd recompute
  (`deepseek.py:_moe`) differentiates the **UN-chunked** combine; the chunked forward is
  never differentiated. Rung 7 flag = chunked-in-remat: the recompute IS differentiated
  through the chunked path (per-chunk `ring_ragged_unsort` vjp + psum_scatter transposes).
- `pl.kernel` API shim: this tree pins **jax 0.10.0** → `out_shape`/`scratch_shapes` kwargs
  (see `ragged_gather_reduce.py:26`); the old-branch kernel uses `out_type`/`scratch_types`
  (jax >0.10). Port must adopt the same `Version(...)` shim.
- Structural extras that must keep working: `moe_shared_after_combine` scheduling token
  ([1,1] slice of chunk-0's **pre-RS** combined output) and the per-chunk barrier chain.

**The swap**: per-chunk `[ring_ragged_unsort → psum_scatter]` → per-chunk
`[fused_combine_rs_kernel(buf_f32, sidx_c, ...) → [tok_per_chunk/EP, H]]`, chunk loop +
concat + (adapted) fences kept. Per-chunk kernel work is O(T/N) by construction (it gathers
only the rows `sidx_c` references; `sidx_c` is the sliced operand).

---

## 2. Gap list

Ordered by how hard each attacks the prize (SC tail 17.3 → ~10 ms).

**G1 — Single-subcore TEC vs the production 16-subcore combine (perf thesis, part 1).**
The fused kernel's TEC mesh is `num_cores=1, num_subcores=1`. The 9.5 ms production combine
runs `ragged_gather_reduce` partitioned across the full SC (2×16 subcores on v7x). A
1-subcore gather-reduce at production shape has **never been timed on v7x**; if its DMA issue
rate / reduce throughput is a large multiple of 9.5 ms, the fusion loses outright even with a
perfectly hidden RS. v5p proof shapes were small. [projected — THE first measurement gate.]
Mitigation if slow: widen the TEC body to multi-subcore (partition token tiles across
subcores like `ragged_gather_reduce` does) — but a multi-subcore TEC co-programmed with the
SCS body is **new kernel R&D**, not a port (mpmd co-programming was only ever proven at 1+1).

**G2 — f32 RS = 2× wire bytes (perf thesis, part 2).**
The kernel is f32-only; the XLA RS it replaces moves bf16. Direct-to-owner RS per device
sends (EP−1)·CH·H·4 B vs the bf16 ring's (EP−1)·CH·H·2 B → **2× ICI bytes** [derived]. If the
SC-offloaded bf16 RS's 7.8 ms is near link limits, the f32 in-kernel RS needs ≈15.6 ms and
cannot hide under 9.5 ms of combine → tail ≈15.6 ms, prize collapses to ~1.7 ms/layer
[derived, worst case]. It only wins if SCS point-to-point DMA achieves much higher link
utilization than the SC-offload RS (plausible — memory records cross-chip collective BW at
only 16–45 GB/s, and `xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3` in the prod
flags suggests the offload RS is known-slow — but UNMEASURED at this shape). Mitigations:
(a) cast `partial` to bf16 before the remote copy, EP-sum in f32 (halves wire bytes, matches
psum_scatter numerics; needs a TEC cast whose SC bf16 store shapes must be checked — kernel
R&D, medium); (b) accept f32 and re-baseline.

**G3 — Invalid-slot row-0 aliasing at EP>1 (correctness bug in the adapter).**
`_fused_cr_inputs` clamps invalid (other-shard) slots to gather row 0 and relies on the
weight pre-fold to zero them — but `w_buf[0] ≠ 0` whenever row 0 is a valid target for some
slot (always true in truncated/packed buffer mode, where every shard's row 0 is local data;
true on shard 0 in global mode). Those slots then double-count row 0's contribution. The old
branch's own docstring flags it ("the kernel/buffer must treat them as zero"); it was only
validated at EP=1 + a masked numpy check. **Fix required before any EP>1 run**: append one
guaranteed-zero row to `buf` (`[BUF+1, H]`) and clamp invalid `sidx` → `BUF`. Cheap, but it
must be in the port, with a unit gate at EP=4 ragged/truncated inputs.

**G4 — A3-bwd: what it is, and whether it's needed here. (Answer: NOT for rung 6.)**
"A3" = the third architecture iteration of the kernel (full-mesh device_id + held-index SMEM
input + H-sub-tiling, commit `1c83f804c`); "A3-bwd" = running the **fused backward kernel**
under that regime — blocked by the scan/custom_vjp constvar wall (SC Pallas in a bwd rule
constvars under `lax.scan`), which is why the final branch shipped the pure-JAX
`linear_transpose` bwd. Under this tree's manbwd (rung 6 default): the chunked forward path
is **forward-only** — the recompute differentiates the un-chunked combine
(`use_chunked_combine=self.config.moe_chunked_combine_in_remat` = False in `deepseek.py:737`)
— so the fused kernel's vjp is **never invoked**. Port the pure-JAX `linear_transpose`
custom_vjp anyway (≈30 LOC, proven, incl. the `sidx`-as-argument tracer fix) as the safety
net and the rung-7 enabler; the fused bwd kernel stays out of scope. Consequence: fwd
(fused kernel, f32 chain) vs bwd (recompute through un-chunked bf16 combine) are not
bit-identical — same class of reduce-order mismatch the rung-6 fwd-only chunking already
accepts; gate with the existing loss-curve A/B, not bit-exactness.

**G5 — XLA SC-offloaded RS "bypass": by construction, plus a contention question.**
The fused path emits **no `psum_scatter` HLO** for this op — there is nothing for
`xla_tpu_enable_sparse_core_collective_offload_reduce_scatter` to pattern-match, so no flag
change is needed and other offloaded collectives (grad RS, weight AG) are untouched. Two real
interactions remain: (a) other SC-offloaded collectives share the SparseCore with the fused
kernel — mpmd kernels act as scheduling barriers (memory: mpmd-map-sc-barrier), so an
offloaded AG landing inside the MoE tail serializes exactly as today (no regression, but no
improvement either — check the xprof); (b) semaphore/queue coexistence of the XLA SC-offload
runtime with a long-running Pallas SC kernel at production duration is unproven — a hang here
looks like the step-0 SC fatal class from memory (MAX_RESTARTS=3, fresh image tag gotchas).

**G6 — Chunk granularity becomes a free design axis (and the N-chunk loop may be vestigial).**
The fused kernel internally pipelines RS-chunk `c` under combine-chunk `c+1` **over EP** — a
single unchunked fused call already hides the RS. Keeping the outer N-chunk loop (per the
brief) preserves the barrier chain + shared-expert fence structure and shrinks per-call HBM
scratch (`recv = EP·CH_chunk·H·4` B), at the cost of N× kernel-launch + init-barrier +
`ready`-handshake overheads (2 cross-device semaphore rounds each) and N× EP-sum tails. Plan:
wire per-chunk first (drop-in), then sweep N∈{1,2,4} — N=1 may win.

**G7 — Scheduling-token semantics change (`moe_shared_after_combine`).**
The [1,1] token is sliced from chunk-0's **pre-RS** combined output; the fused kernel exposes
no pre-RS tensor. Nearest equivalent: slice chunk-0's fused OUTPUT (post-RS) — the shared
expert then fences on chunk-0's combine+RS rather than combine-only, narrowing its overlap
window by one RS. Acceptable for N≥2; document, re-profile, and if it matters use the
`scatter=False` token trick is NOT available (changes numerics) — accept or fence on `sidx`.

**G8 — f32 pre-fold materialization.**
`op_w = (interm * w_buf[:,None]).astype(f32)` materializes a full `[BUF, H]` f32 buffer
(2× the bf16 GMM output's bytes; at CH=16384·EP tokens ·K=8 this is multi-GB HBM + write
traffic [derived — size needs config receipt]) plus a `[BUF]` scatter-add for `w_buf`. TC is
idle in this window (profile forensics) so the time cost is likely small, but HBM headroom
must be checked in the AOT memory dump. Built ONCE per layer (buffer rows partition across
chunks — `revert` is a permutation), not per chunk.

**G9 — v7x/versioning mechanics (low risk, checklist).**
NL=16 read from `get_tpu_info` (never hardcoded — asserts pass for K=8, TT=16); per-chunk
`CH_chunk = tokens/(N·EP)` must divide by TT=16 (config assert); jax **0.10.0** pin → adopt
the `out_shape`/`scratch_shapes` + `_COMPILER_PARAMS` shim from `ragged_gather_reduce.py`
(the old branch's `needs_layout_passes=False` param must be checked against 0.10.0's
CompilerParams); EP=4 → X-axis on the 4×8×8 torus (playbook: `create_device_mesh`, already
how prod runs); 2 cores/chip share one SC — per-device SC kernels already run in prod, but
BOTH devices of a chip will now run the fused kernel concurrently on the shared SC
(same as today's combine kernels — no new exposure, worth an xprof glance).

**G10 — Remat plumbing largely NOT needed here.**
The old branch's decoders.py `checkpoint_name` save-plumbing existed so autodiff remat would
not re-run the SC kernel. Under manbwd the recompute takes the un-chunked path, so the fused
kernel is never re-run in bwd. Keep the `checkpoint_name` tag (harmless, enables non-manbwd
configs); skip the decoders.py policy edits unless a non-manbwd config is A/B'd.

---

## 3. Risk-ranked port plan with validation gates

CPU-viability for the SC kernel itself: **none** (correctness/perf need hardware). The two
non-hardware gates that DO exist: local AOT Mosaic compile (virtual tpu7x topology, catches
every shape/tiling/VMEM error, ~2–3 min) and the CPU equivalence test of the ADAPTER math
(pure-JAX reference of pre-fold + zero-row + permute vs `chunked_ring_combine_reduce_scatter`
fallback path). Hardware rungs: **bodaborg-tpu7x-sps** (single-host v7x, ConfigMap inject —
verify ≥4 chips so the EP=4 cross-chip ring is real; fall back to the 2×2×1 xpk node from the
tstat drills or v5p if not) → 4×8×8 cluster.

**Step 0 — Kill/confirm the perf thesis BEFORE porting anything (1 bodaborg cycle).**
Standalone script (drills-style, no MaxText): production per-chunk shape
(CH_chunk·K slots, H=7168, EP=4, f32), three timings on the SC lane via xprof:
(i) fused kernel `scatter=False` (TEC-only) vs (ii) the in-repo `ragged_gather_reduce` at the
same shape (the 9.5 ms proxy) — resolves **G1**; (iii) fused `scatter=True` minus (i) → SCS
RS exposure at f32 — resolves **G2** (compare against 7.8 ms × chunk-fraction). 
GATE: projected fused tail < ~0.8× current 17.3 ms chain ⇒ GO. Else STOP and pivot
(multi-subcore TEC R&D or bf16-partial variant) before any integration.

**Step 1 — Port the kernel file (0 hardware cycles).**
Copy `fused_combine_rs.py` → `src/maxtext/kernels/fused_combine_rs.py`; apply the jax-0.10
kwarg shim; add the zero-row contract note. GATE: local AOT Mosaic compile
(`topologies.get_topology_desc("tpu7x:4x4x4")`, production CH_chunk/H/K, full 12-axis-style
mesh with held axes) — must exercise **production token counts** (the NTOK=64 harness
false-pass is the documented trap).

**Step 2 — Adapter + zero-row fix + numerics (1–2 bodaborg cycles).**
Port `_fused_cr_inputs` with: (a) the `[BUF+1,H]` zero-row + invalid→BUF clamp (G3); (b)
per-chunk slicing after the chunk-major permute (sidx_c from `ridx_c`, global `w_buf` built
once); (c) both buffering modes (global + truncated). GATE A (CPU): adapter-math unit test vs
the pure-JAX reference (`jnp.take`+K-sum+psum_scatter) on the enforce-fallback path, ragged
`group_sizes`, EP=4, both modes — extends `tests/.../equiv_chunk_test.py`. GATE B (bodaborg):
fused-vs-chunked forward equivalence at EP=4 with REAL ragged routing (not all-ones — the
gmm_v2-init memory says degenerate routing hides/creates overflow classes), rel < 2e-2,
plus the truncated-buffer mode.

**Step 3 — Wire into the chunked path (0 hardware cycles).**
New flag `moe_fused_combine_rs` (types.py + base.yml); branch inside
`chunked_ring_combine_reduce_scatter` (or a sibling function) replacing per-chunk
unsort+psum_scatter; keep barrier chain (fence on previous chunk's fused OUTPUT); post-RS
scheduling token (G7); pure-JAX `linear_transpose` custom_vjp with `sidx`-as-arg (G4);
`checkpoint_name` tag. GATE: local AOT `train_compile` HLO dump of the full model
(memory: local-aot-hlo-dump, ~3 min) — compiles, no SC allocation failure, HBM within
budget (G8), and the tail region shows the fused kernel with NO psum_scatter (G5 check);
plus the tiny-config CPU real-jit to catch tracer escape (memory: aot-misses-tracer-escape).

**Step 4 — Cluster: sanity then honest A/B (2–3 cluster cycles).**
(commit → docker build/push fresh tag → xpk; MAX_RESTARTS=3 for SC-fatal flake.)
Cycle 4a: step-0 + 20-step run, fused vs rung6 baseline loss curves overlaid (reduce-order
tolerance, not bit-exact), watch for SC hangs (G5b). Cycle 4b: xprof A/B — the ONLY honest
proof is SCS remote-DMAs visibly concurrent with TEC gather-reduce on the SC lane and the
17.3 ms chain shortened; wall-clock arithmetic alone already produced one retracted claim
(83–86% hide). Cycle 4c (optional): N∈{1,2,4} sweep (G6) + `moe_shared_after_combine`
re-check (G7).

---

## 4. Estimate

| Item | Estimate |
|---|---|
| Files touched | 6–7: `kernels/fused_combine_rs.py` (new, ~420 LOC ported), `kernels/ragged/ragged_sort.py` (~50), `layers/moe.py` (~120: adapter + vjp + branch), `configs/types.py`+`base.yml` (~35), `tests/.../equiv_chunk_test.py` extension (~120), standalone step-0 bench script (~120, lives in perf-drills) |
| Total new/ported LOC | ~850, of which ~60–70% is verbatim port |
| Hardware cycles to first honest A/B | **~6**: 1 bodaborg (step-0 thesis bench) + 1–2 bodaborg (numerics) + 2–3 cluster (sanity + profile A/B). Local AOT gates between each cost no hardware. |
| Schedule shape | Step 0 is front-loaded and can kill the project in one cycle — do it before writing a line of integration code. |

---

## 5. The biggest risk, bluntly

**The perf thesis is unproven at exactly the two points that determine the prize, and both
cut against it.** (1) The ported kernel's combine runs on ONE vector subcore; the 9.5 ms
baseline it must match runs on all 32 (2×16) — the kernel has never been timed at production
shape on v7x (the only production attempt died at compile pre-token-tiling, and the harness
"pass" was at NTOK=64, 256× under real size). (2) The kernel is f32-only, so its in-kernel RS
moves 2× the bytes of the bf16 RS it replaces — if the 7.8 ms offload RS is anywhere near
link-limited, the "hidden" RS needs ~15.6 ms of DMA under 9.5 ms of compute and is
arithmetically un-hideable. Either failure alone reduces the port to a wash; fixing them
(multi-subcore co-programmed TEC, bf16 partials on the wire) is new kernel R&D, not porting.
Everything else — the EP>1 row-0 aliasing bug, the bwd story, flag interactions — is known,
bounded, and cheap. Hence the plan's shape: **one standalone bodaborg measurement (Step 0)
before any integration work.** The execution risks (v7x cross-chip SCS DMA at production
duration coexisting with the XLA SC-offload runtime) are real but testable in the same cycle.

---

## 6. STEP-0 RESULT (2026-07-03): thesis DEAD as-is — measured on v7x hardware

Bench executed on `sivaibhav-exp-v7x` single-host v7x 2x2x1 (8 cores; bodaborg-tpu7x-sps is
down — its reservation has failed `CONDITION_NOT_MET` since 2026-06-25, job left armed there).
Mesh (expert=4, fsdp=2): EP=4 cross-chip ring, both cores/chip active. Kernel = old-branch
`fused_combine_rs.py` @ `e03ab077a` verbatim. Full receipt: `FCR_STEP0_BENCH_LOG.txt` (this
dir). jax 0.10.1, `SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16)`.

All numbers below **[measured]**, ms/iter over 10 iters, per shape `S`=slots/device, H=7168, K=8, EP=4:

| | S=65536 (CH=2048) | S=131072 (CH=4096) |
|---|---|---|
| fused TEC-only (`scatter=False`) | **117.4** | **234.8** |
| fused FULL (`scatter=True`) | 123.0 | 245.9 |
| `ragged_gather_reduce` f32 all-valid (same inputs) | **5.14** | **10.31** |
| `ragged_gather_reduce` f32 ~1/EP-valid | 1.54 | 2.99 |
| `ragged_gather_reduce` bf16 ~1/EP-valid | 1.60 | 3.09 |
| `psum_scatter` f32 / bf16 (this fabric) | 1.97 / 1.14 | 3.86 / 2.20 |
| correctness vs jnp ref | rel=1.5e-07 ✓ | **rel=9.3e-01 ✗ WRONG** |

1. **1-subcore vs 32-subcore: 22.9× / 22.8× slower** at identical all-valid f32 inputs. The
   fused TEC gathers at a flat **16 GB/s** (1-subcore DMA-issue-bound; shape-independent).
   G1 confirmed at full magnitude.
2. **RS hide**: exposure (FULL − TEC) = 5.6 / 11.1 ms — the RS "hides" only because the combine
   is 23× too slow (an enormous window). Even so, the SCS-driven f32 RS exposes MORE than the
   entire TC psum_scatter_f32 costs on this fabric.
3. **Projected production tail** [derived]: production combine chain is 9.5 ms on 32 subcores ⇒
   1-subcore fused ≈ 9.5 × ~23 ≈ **~220 ms** vs the 17.3 ms it must beat. Same-fabric honest
   chain comparison: fused FULL 123 ms vs unfused (rgr bf16 qv + psum bf16) **2.7 ms** = 45×.
4. **Bonus blocker**: the kernel is numerically WRONG at CH=4096 (fine at CH=2048) — a latent
   shape-dependent bug (suspect the 2-deep `partial_ref` double-buffer / `scs_to_tec`
   pipelining); the v5p proofs and the NTOK=64 v7x harness never exercised this CH.

**VERDICT: do not port this kernel.** The specific R&D that would revive the idea is NOT
porting/optimizing the drill kernel but the inverse composition: add an SCS reduce-scatter
body to the production `ragged_gather_reduce` (co-program its existing 32-subcore vector body
with a scalar-subcore RS streamer). Unproven: mpmd co-programming was only ever validated at
1 core + 1 subcore, and rgr's row/column partitioning must be reconciled with per-owner-chunk
streaming order. That is a kernel-track project, not a MaxText port.

**Cheaper next lever (recommend benching next): move the production RS off the SC queue onto
the TC lane.** The 7.8 ms problem is that the SC-offloaded RS serializes behind the mpmd
combine on the SAME engine. `psum_scatter` on the TC lane measured 1.1–2.2 ms for this volume
class here, and the old branch already carries a validated TC-Pallas direct-to-owner RS with
custom_vjp (`moe_direct_rs`, == psum_scatter rel 0.004, v5p). Disabling SC-offload for this
one RS (or flipping `moe_direct_rs` on) lets it overlap the SC combine on a different engine —
zero new kernel work. Needs a production-shape 4x8x8 profile to confirm the TC lane has room.
