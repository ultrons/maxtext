# DeepSeek-V3 671B — v7x training perf, 14.50 s/step

Reproduction of the **14.50 s/step** configuration for DeepSeek-V3 671B pretraining on
a TPU v7x `4x8x8` slice (256 chips / 512 cores). All numbers below are measured on that
hardware; loss is bit-exact to the reference curve (`12.270` @ step 1 → `9.324` @ step 19).

Branch HEAD: `85f6ad2d3`. Model: `deepseek3-671b`, PBS 4, seq 4096, mesh FSDP=128 × EP=4.
Data: synthetic. Steps: 20 (profiler at 6–8). Loss acceptance bar: **digit-exact** vs the curve.

## The optimization ladder (each step measured, loss digit-exact)

| Config | s/step | Δ | What changed |
|---|---|---|---|
| rung-4 baseline | 15.33 | — | `moe_n_chunks=2` + handwritten bwd stack |
| rung-8b | 15.22 | −0.11 | combine-bwd: N per-chunk cotangent AGs → 1 tiled AG (`522d59086`) |
| + splash host-offload | 15.45* | — | save splash O+LSE to pinned host, delete fwd recompute (`20b2cdf21`) |
| + LHS async-depth flag | 14.64 | −0.81 vs above | `--xla_lhs_prioritize_async_depth_over_stall=true` (unlocks the offload) |
| + multi-SC AG offload | **14.50** | −0.14 | `--xla_tpu_use_single_sparse_core_for_all_gather_offload=false` |

\* splash-offload alone regressed (+0.24) because the deleted recompute exposed a backward
all-gather; the LHS async-depth flag is what converts it to a win. The two must ship together.

### Why the last two flags matter (measured)
- **`xla_lhs_prioritize_async_depth_over_stall=true`** — the single carrier of the −0.81
  splash-offload win. The latency-hiding scheduler otherwise issues the pinned-host O/LSE
  restores too late; this makes it prefetch them a layer early so they overlap backward compute.
  Null without host-offload (needs the async work + freed compute to exploit).
- **`xla_tpu_use_single_sparse_core_for_all_gather_offload=false`** — lets offloaded
  all-gathers use both SparseCores instead of piling onto SC-0. The exposed combine-bwd
  cotangent AG and a weight re-gather AG were colliding on one SC; splitting them across the
  two SCs lets them run in parallel (the weight AG hides under the combine AG). The combine AG
  itself is unchanged (~9.4 ms); the win is the parallelism.

## Build the image

```bash
# from this branch (rung6-sliced / ds-v3-perf-14.50)
sudo docker build -f Dockerfile.baseline -t <registry>/maxtext:ds-v3-14.50 .
sudo docker push <registry>/maxtext:ds-v3-14.50
```

## LIBTPU_INIT_ARGS (the XLA flags — the two perf-critical ones are marked)

```
--xla_tpu_dvfs_p_state=7
--xla_tpu_scoped_vmem_limit_kib=65472
--xla_tpu_bf16_emission_mode=NATIVE_EMISSION
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_all_gather_offload_tracing=true
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
--xla_enable_async_all_gather=true
--xla_tpu_prefer_async_allgather_to_allreduce=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_use_single_sparse_core_for_all_gather_offload=false   # PERF: multi-SC AG offload (−0.14)
--xla_tpu_enable_concurrent_sparse_core_offloading=true
--xla_tpu_aggressive_opt_barrier_removal=true
--xla_tpu_enable_offloading_gather_to_sparsecore=true
--xla_tpu_sparse_core_all_gather_latency_multiplier=1
--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true
--xla_tpu_pcie_bandwidth_multiplier=0.03
--xla_lhs_prioritize_async_depth_over_stall=true               # PERF: unlocks host-offload (−0.81)
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
--xla_tpu_enable_3d_reduce_scatter_decomposer=false
```

## Launch command (MaxText flags)

The MoE-feature flags that define this config are grouped at the end.

```bash
export LIBTPU_INIT_ARGS='<the args above, space-separated>'
export JAX_PLATFORMS='tpu,cpu'

python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml \
  model_name=deepseek3-671b per_device_batch_size=4.0 max_target_length=4096 \
  ici_fsdp_parallelism=128 ici_expert_parallelism=4 ici_data_parallelism=1 \
  ici_fsdp_transpose_parallelism=1 dcn_data_parallelism=-1 dcn_pipeline_parallelism=1 \
  ici_pipeline_parallelism=1 shard_exp_on_fsdp=False use_iota_embed=True \
  tokenizer_path=assets/tokenizer.mistral-v3 dataset_type=synthetic \
  opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 dtype=bfloat16 \
  sa_use_fused_bwd_kernel=True megablox=True sparse_matmul=True use_tokamax_gmm=True \
  use_gmm_v2=True use_tokamax_splash=True use_max_logit_estimate=-1 \
  cost_estimate_flops_fwd=5000000000000 cost_estimate_flops_bwd=5000000000000 \
  float32_weight_sum=False remat_policy=custom allow_split_physical_axes=False \
  decoder_layer_input=device attention=flash \
  sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
  sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
  sa_block_kv_dq=2048 sa_block_q_dq=2048 \
  use_random_routing=True use_ring_of_experts=True use_custom_sort_vjp=True use_ragged_sort=True \
  merge_gating_gmm=False \
  wi_tile_fwd_batch_seq=256 wi_tile_fwd_embed_dim=7168 wi_tile_fwd_mlp_dim=1024 \
  wi_tile_dlhs_batch_seq=256 wi_tile_dlhs_embed_dim=3584 wi_tile_dlhs_mlp_dim=2048 \
  wi_tile_drhs_batch_seq=512 wi_tile_drhs_embed_dim=1792 wi_tile_drhs_mlp_dim=2048 \
  wo_tile_fwd_batch_seq=512 wo_tile_fwd_embed_dim=3584 wo_tile_fwd_mlp_dim=2048 \
  wo_tile_dlhs_batch_seq=512 wo_tile_dlhs_embed_dim=1792 wo_tile_dlhs_mlp_dim=2048 \
  wo_tile_drhs_batch_seq=512 wo_tile_drhs_embed_dim=1792 wo_tile_drhs_mlp_dim=2048 \
  skip_jax_distributed_system=True steps=20 profiler=xplane \
  skip_first_n_steps_for_profiler=5 profiler_steps=3 \
  moe_n_chunks=2 \
  decouple_combine_rs_chunks=4 \
  moe_chunked_combine_in_remat=True \
  moe_splash_host_offload=True \
  moe_direct_rs=True \
  moe_routing_key_as_input=True \
  moe_handwritten_bwd=True \
  moe_weight_ag_scheduling_group=True
```

## The MoE feature flags (what this branch adds), in dependency order

| Flag | Effect |
|---|---|
| `moe_handwritten_bwd=True` | hand-written layer backward (gather + attention + MoE); prerequisite for the rest |
| `moe_weight_ag_scheduling_group=True` | scheduling-group tag so the handwritten bwd doesn't blow up the scheduler |
| `moe_routing_key_as_input=True` | thread routing key through as input (needed by the handwritten bwd) |
| `moe_n_chunks=2` | ring-of-experts: split tokens into 2 chunks; frees cross-engine RS hiding under GMMs |
| `decouple_combine_rs_chunks=4` | decouple the combine→reduce-scatter chunking from the GMM chunking |
| `moe_chunked_combine_in_remat=True` | run the chunked combine inside the remat region (memory-flat bwd) |
| `moe_direct_rs=True` | TC-Pallas direct-to-owner reduce-scatter (== psum_scatter), off the SC queue |
| `moe_splash_host_offload=True` | save splash O+LSE to pinned host in fwd, restore in bwd; deletes the splash recompute |

## Expected result

Steady-state (steps 11–19): **~14.50 s/step**. Loss must be **digit-exact**:
`12.270` (step 1) → `9.987` (11) → `9.324` (19). Any deviation is a numerics bug, not noise.

See `EXPERIMENT_LOG.md` for the full measurement history, the levers that were tried and
retired (with receipts), and the two open levers (making the combine-bwd AG itself faster,
and decoupling forward/backward GMM chunking).
