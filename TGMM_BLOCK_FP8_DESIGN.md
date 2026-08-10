# Block-scaled fp8 tgmm (wgrad) — Task-2 design (grounded, ready to build)

Goal: NVIDIA-analog fine-block fp8 wgrad. dW[k,n] = Σ_m x_sorted[m,k]·δ[m,n], BOTH operands e4m3
with block-local dynamic scales along the contracting m dim. Numerical gate banked
(/mnt/disks/scratch/test_block_tgmm.py): block-256 e4m3 grad rel=9.3e-3 vs per-tensor e5m2 4.3e-2
(4.7x); both-operand block e4m3 rel=2.8e-2; per-tensor e4m3 2.2e-2.

## Key kernel facts (from vendored /mnt/disks/scratch/tgmm_vendor/tgmm_kernel_orig.py, 779 lines,
   = tokamax pallas_mosaic_tpu_v2_tgmm_kernel.py)
- grid=(num_n, num_k, num_gm); gm = group-major tiles along m, GROUP-ALIGNED with DYNAMIC offsets
  (`gm_id_to_m_offset` SMEM, computed IN-KERNEL in tgmm_kernel_main from scalar-prefetched
  group_sizes; max_num_gm = size_group + cdiv(size_m, tile_m) - 1).
- `tgmm_inner_kernel._matmul`: masks tile rows to [m_start,m_end), f32 dot, accumulates in
  acc_ref across gm steps of a group; rhs per-N scale applied ONLY at group-change writeout.
- Sub-channel scale ((m_blocks,1,n), leading dim>1) is ANTICIPATED but rejected at
  make_tgmm_configs lines 210-215 ("not implemented"). lhs has NO quant path (line 92).

## Design decision: quant blocks = the gm SEGMENTS (not fixed-256 blocks)
Fixed-stride blocks cross the dynamic gm boundaries -> in-kernel per-fixed-block dequant would need
dynamic splits of the dot. Instead quantize per gm segment: each gm tile then has EXACTLY ONE
static scale row. The MXU >=256-contracting-block constraint does NOT apply: the dot runs raw
e4m3 x e4m3 -> f32; scales apply POST-dot per tile (VPU), so segment length is unconstrained.
Empty/short segments: scale guard +1e-30.

## Build plan
1. New module src/maxtext/kernels/tgmm_block.py, vendoring from tgmm_kernel_orig.py:
   entry `tgmm_block_fp8(lhs, rhs, group_sizes, num_actual_groups, group_offset, tile_info, ...)`,
   drop-in for tokamax tgmm_v2 (same out [G,K,N]), taking BF16 lhs/rhs and quantizing INSIDE the
   entry (caller-side jnp, before pallas_call):
   - Replicate the gm decomposition in jnp from (group_sizes, tile_m): per group g with size s_g,
     split at tile_m strides -> gm segment boundaries; TRANSCRIBE the exact rule from
     tgmm_kernel_main's metadata fill (READ IT FIRST — not yet read; it fills
     gm_id_to_group_id/gm_id_to_m_offset). Must match bit-for-bit; factor as one shared helper +
     a property test vs the transcribed rule.
   - seg_id[m] = searchsorted(gm_offsets, m, 'right')-1; sx = segment_max(|x|)/448 + 1e-30
     -> [max_num_gm, K]; sd likewise [max_num_gm, N]; qx/qd = clip(x/s, +-448) e4m3.
2. Kernel deltas:
   - lhs becomes OperandRef(value, scale) like rhs (or 2 extra operands); BlockSpecs:
     lhs_scale (1, tile_k) index (gm_id -> (gm_id, k_id)); rhs_scale (1,1,tile_n) index
     (gm_id, 0, n_id) — CHANGED from (0,0,n_id); scale arrays [max_num_gm, K]/[max_num_gm,1,N]
     (f32, N padded to aligned_n as the entry already does for rhs_scale).
   - `_matmul`: partial = dot(lhs_masked_e4m3, rhs_masked_e4m3, f32);
     partial *= ls[0].reshape(-1,1) * rs[0]  (EVERY gm step, before accumulation);
     REMOVE the has_scale group-change multiply at writeout.
   - make_tgmm_configs: allow the new mode (bypass the 210-215 rejection via a flag/new configs
     field); has_scale semantics -> per-gm.
   - e4m3 operands: sublane tiling for 1-byte dtypes comes from get_sublane_tiling (check
     size_lhs_sublane usage; the DMA reshape (-1, sublane, k) must use the fp8 sublane count).
3. ops.py wiring (flag `use_block_fp8_tgmm`, default off): in `_compute_drhs` /
   `_drhs_run_tokamax_v2` call tgmm_block_fp8 with the RAW drhs_dout (skip its bf16
   normalization/quantize) + the residual lhs (x_sorted, bf16). Downstream unchanged (my
   _gmm_bwd QArray wrap consumes drhs AFTER; output dtype still bf16 wire).
4. Numerics gate: interpret-mode CPU (tgmm has NO ICI -> interpret works): vendored kernel vs a
   jnp segment-block reference (adapt test_block_tgmm.py from fixed-256 to gm-segment blocks) —
   want exact match of the emulation (same math), and rel vs bf16 reference ~1e-2 class.
5. AOT gate on the record config + flag: compiles; tgmm custom-calls show e4m3 operands;
   temps sane; no structural regressions (standard census).

## Perf/context notes
- tgmm labels ~160-240ms/step TC; fp8 ~1.5x on memory-bound parts; value = accuracy-preserving
  fp8 grads (NVIDIA arXiv:2506.08027: e5m2 degrades under block scaling, all-e4m3 matches bf16).
- Composes with bwd_quantization_dtype=e4m3 (Task 1, commit 4b7654be5, cluster arm
  siv-cn-e4m3bwd) — that flag covers the qwix dot_general/gmm bwd quantize; THIS covers the tgmm
  that we bf16-normalized.
- Collective ids in use: 0,7+chunk,40,45,50,55,56 (tgmm needs none — no ICI).
