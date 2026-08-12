# DeepSeek-V3 671B — v7x training perf, 14.10 s/step (record)

Reproduction of the **14.10 s/step** config on a TPU v7x `4x8x8` slice (256 chips / 512 cores),
deepseek3-671b, PBS 4, seq 4096, FSDP=128 × EP=4, synthetic data. Loss bit-exact to the reference
curve (12.270 @ step1 → 9.324 @ step19). All numbers same-image measured A/B (no projections).

## The ladder (each measured, loss digit-exact)
| config | s/step | lever |
|---|---:|---|
| rung-4 baseline | 15.33 | moe_n_chunks=2 + handwritten bwd stack |
| + splash host-offload + LHS async-depth + multi-SC AG | 14.50 | see the 14.50 branch (`ds-v3-perf-14.50`) |
| + moe_direct_token_ag | 14.33 | dispatch token-AG → TensorCore (off the SC offload queue), −0.20 |
| + moe_direct_combine_ag | ~14.20 | combine-cotangent AG (.626) → TensorCore, −0.10 |
| + qk_diag_skip | **14.10** | splash fused fwd+bwd causal-diagonal QK skip (bit-exact), −0.05; TOTAL −0.43 (super-additive) |

## The insight
**Exposed-collective PLACEMENT is the lever, not compute FLOPs.** Moving the two SparseCore-offloaded
EP all-gathers (backward dispatch token-AG and combine-cotangent AG) onto the TensorCore — via a
direct-to-owner Pallas kernel (`_direct_all_gather`, the transpose of `moe_direct_rs`) — gets them off
the serialized single-SC queue so XLA overlaps their ICI DMAs with the SC combine/gather work. This
stacked super-additively (−0.20 + −0.10 measured as −0.43 combined). Compute-kernel FLOP wins
(qk_diag's diagonal skip, −14.6% fwd / −5.9% bwd kernel) got overlap-absorbed to ~−0.05 step.
Scheduling-group co-tags in the backward all hit walls (cycle or serialize) — placement wins, ordering loses.

## Build + run
Build **`Dockerfile.qkdiag`** (baseline + the `docker/tokamax_splash_attention_kernel.py` overlay for
qk_diag). For the flag-off control, build `Dockerfile.baseline`.

The 14.10 config = the canonical 14.50 flag/LIBTPU set (see `EXPERIMENT_LOG.md`, incl.
`--xla_lhs_prioritize_async_depth_over_stall=true`, `--xla_tpu_use_single_sparse_core_for_all_gather_offload=false`,
`moe_splash_host_offload`, `moe_direct_rs`, `moe_handwritten_bwd`, `decouple_combine_rs_chunks=4`,
`moe_chunked_combine_in_remat`, `moe_weight_ag_scheduling_group`, `moe_routing_key_as_input`,
`moe_n_chunks=2`) **plus these four flags**:

```
moe_direct_token_ag=true      # dispatch token-AG → TC   (collective_id 40)
moe_direct_combine_ag=true    # combine-cotangent AG → TC (collective_id 50)
qk_diag_skip=true             # splash fwd+bwd diagonal QK skip (bit-exact)
qk_diag_grid=4                # sub-grid granularity (sweet spot)
```

All four default false → the OFF control needs no flag changes.

## Expected
Steady-state (steps 11–19) ~**14.10 s/step**; loss digit-exact 12.270 → 9.324. Any loss deviation = a
numerics bug (all four levers are bit-exact).
