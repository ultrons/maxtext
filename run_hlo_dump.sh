#!/bin/bash
cd /mnt/disks/scratch/maxtext-1410-upstream
mkdir -p /mnt/disks/scratch/hlodump_a2a; rm -f /mnt/disks/scratch/hlodump_a2a/* 2>/dev/null
BASEFLAGS=$(grep -vE '^\s*(#|$)' flags_record_1410.txt | sed 's/[[:space:]]*$//' | tr '\n' ' ')
A2A_ARGS="model_name=deepseek3-671b per_device_batch_size=1.0 max_target_length=4096 \
ici_fsdp_parallelism=32 ici_expert_parallelism=16 ici_data_parallelism=1 ici_fsdp_transpose_parallelism=1 \
dcn_data_parallelism=-1 dcn_pipeline_parallelism=1 ici_pipeline_parallelism=1 \
shard_exp_on_fsdp=False use_iota_embed=True tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic dataset_path=gs://max-datasets-rogue opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 dtype=bfloat16 \
sa_use_fused_bwd_kernel=True megablox=True sparse_matmul=True use_tokamax_gmm=True use_gmm_v2=True use_tokamax_splash=True \
float32_weight_sum=False remat_policy=custom allow_split_physical_axes=True decoder_layer_input=device \
async_checkpointing=False enable_checkpointing=False attention=flash \
use_random_routing=True use_ring_of_experts=False use_custom_sort_vjp=True use_ragged_sort=True merge_gating_gmm=False \
optimizer_memory_host_offload=True ragged_buffer_factor=2 skip_jax_distributed_system=True steps=20"
COMPILE="compile_topology=tpu7x-512 compile_topology_num_slices=1 base_output_directory=/tmp/aot run_name=hlo_a2a"
sudo docker run --rm \
  -v /mnt/disks/scratch/maxtext-1410-upstream/src/maxtext:/deps/src/maxtext:ro \
  -v /mnt/disks/scratch/hlodump_a2a:/hlodump \
  -e LIBTPU_INIT_ARGS="$BASEFLAGS" \
  -e XLA_FLAGS="--xla_dump_to=/hlodump --xla_dump_hlo_as_text" \
  -e JAX_PLATFORMS=cpu -e ENABLE_PJRT_COMPATIBILITY=true \
  gcr.io/tpu-vm-gke-testing/integrate_v2:1410-up2-d59a6a618 \
  python3 -m maxtext.trainers.pre_train.train_compile maxtext/configs/base.yml $A2A_ARGS $COMPILE
echo "DUMP_DONE exit=$?"
