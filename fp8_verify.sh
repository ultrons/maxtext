#!/bin/bash
# Verify fp8_full actually fp8s the MoE GMM: AOT-compile + dump optimized HLO + grep for e4m3 GMM operands.
cd /mnt/disks/scratch/maxtext-1410-upstream
mkdir -p /mnt/disks/scratch/hlodump_fp8; rm -f /mnt/disks/scratch/hlodump_fp8/* 2>/dev/null
XLAFLAGS="--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_enable_concurrent_sparse_core_offloading=true"
MT="model_name=deepseek3-671b dtype=bfloat16 per_device_batch_size=1.0 ragged_buffer_factor=2.0 num_moe_token_chunks=2 abort_on_nan_loss=false abort_on_inf_loss=false async_checkpointing=false enable_checkpointing=false remat_policy=custom allow_split_physical_axes=false use_iota_embed=True context=device out_proj=device custom_mesh_and_rule=ep-as-dp decoder_layer_input=device mu_dtype=bfloat16 grad_dtype=bfloat16 dcn_pipeline_parallelism=1 dcn_data_parallelism=-1 ici_pipeline_parallelism=1 ici_fsdp_transpose_parallelism=1 ici_fsdp_parallelism=128 ici_expert_parallelism=8 ici_data_parallelism=1 shard_exp_on_fsdp=false use_custom_sort_vjp=true dataset_type=synthetic dataset_path=gs://max-datasets-rogue opt_type=adamw steps=20 sa_use_fused_bwd_kernel=true use_max_logit_estimate=-1 cost_estimate_flops_fwd=5000000000000 cost_estimate_flops_bwd=5000000000000 float32_weight_sum=False megablox=true sparse_matmul=true use_gmm_v2=true use_tokamax_splash=true use_tokamax_gmm=true attention=flash max_target_length=4096 use_random_routing=true use_ring_of_experts=true use_ragged_sort=true tokenizer_path=assets/tokenizer.mistral-v3 merge_gating_gmm=false skip_jax_distributed_system=true quantization=fp8_full use_qwix_quantization=true"
COMPILE="compile_topology=tpu7x-1024 compile_topology_num_slices=1 base_output_directory=/tmp/aot run_name=fp8verify"
sudo docker run --rm \
  -v /mnt/disks/scratch/maxtext-1410-upstream/src/maxtext:/deps/src/maxtext:ro \
  -v /mnt/disks/scratch/hlodump_fp8:/hlodump \
  -e LIBTPU_INIT_ARGS="$XLAFLAGS" \
  -e XLA_FLAGS="--xla_dump_to=/hlodump --xla_dump_hlo_as_text" \
  -e JAX_PLATFORMS=cpu -e ENABLE_PJRT_COMPATIBILITY=true \
  gcr.io/tpu-vm-gke-testing/integrate_v2:1410-up2-d59a6a618 \
  python3 -m maxtext.trainers.pre_train.train_compile maxtext/configs/base.yml $MT $COMPILE 2>&1 | tail -20
echo "COMPILE_EXIT=${PIPESTATUS[0]}"
