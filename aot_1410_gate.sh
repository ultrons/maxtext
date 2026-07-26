#!/bin/bash
# AOT gate for the rebased 14.10 stack (perf-1410-upstream) at the EXACT launch config:
# full 61-layer deepseek3-671b, FSDP128 x EP4 = 512 cores = 256 chips = 4x8x8, PBS4, seq4096,
# ALL levers ON, num_moe_token_chunks=${CHUNK:-2}. Compiles fwd+bwd on virtual tpu7x-512, CPU host.
set -e
set -o pipefail
IMAGE=gcr.io/tpu-prod-env-automated/integrate_v2:1410-upstream-ac8cfb86a
CHUNK="${CHUNK:-2}"
BASEFLAGS=$(grep -vE '^\s*(#|$)' flags_record_1410.txt | sed 's/[[:space:]]*$//' | tr '\n' ' ')

MODEL_ARGS="\
model_name=deepseek3-671b \
per_device_batch_size=4.0 max_target_length=4096 \
ici_fsdp_parallelism=128 ici_expert_parallelism=4 ici_data_parallelism=1 \
ici_fsdp_transpose_parallelism=1 dcn_data_parallelism=-1 dcn_pipeline_parallelism=1 ici_pipeline_parallelism=1 \
shard_exp_on_fsdp=False use_iota_embed=True tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic dataset_path=gs://max-datasets-rogue \
opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 dtype=bfloat16 \
sa_use_fused_bwd_kernel=True megablox=True sparse_matmul=True use_tokamax_gmm=True use_gmm_v2=True use_tokamax_splash=True \
use_max_logit_estimate=-1 cost_estimate_flops_fwd=5000000000000 cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False remat_policy=custom allow_split_physical_axes=False decoder_layer_input=device \
enable_tpu_profiling_options=True async_checkpointing=False enable_checkpointing=False \
attention=flash sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_kv_dq=2048 sa_block_q_dq=2048 \
use_random_routing=True use_ring_of_experts=True use_custom_sort_vjp=True use_ragged_sort=True merge_gating_gmm=False \
wi_tile_fwd_batch_seq=256 wi_tile_fwd_embed_dim=7168 wi_tile_fwd_mlp_dim=1024 \
wi_tile_dlhs_batch_seq=256 wi_tile_dlhs_embed_dim=3584 wi_tile_dlhs_mlp_dim=2048 \
wi_tile_drhs_batch_seq=512 wi_tile_drhs_embed_dim=1792 wi_tile_drhs_mlp_dim=2048 \
wo_tile_fwd_batch_seq=512 wo_tile_fwd_embed_dim=3584 wo_tile_fwd_mlp_dim=2048 \
wo_tile_dlhs_batch_seq=512 wo_tile_dlhs_embed_dim=1792 wo_tile_dlhs_mlp_dim=2048 \
wo_tile_drhs_batch_seq=512 wo_tile_drhs_embed_dim=1792 wo_tile_drhs_mlp_dim=2048 \
optimizer_memory_host_offload=True skip_jax_distributed_system=True steps=20 \
moe_weight_ag_scheduling_group=True moe_routing_key_as_input=True moe_handwritten_bwd=True \
moe_splash_host_offload=True num_moe_token_chunks=${CHUNK} decouple_combine_rs_chunks=4 \
moe_chunked_combine_in_remat=True moe_direct_rs=True moe_direct_token_ag=True moe_direct_combine_ag=True \
qk_diag_skip=True qk_diag_grid=4"

COMPILE="compile_topology=tpu7x-512 compile_topology_num_slices=1 base_output_directory=/tmp/aot_out run_name=aot_1410_chunk${CHUNK}"

echo "=== AOT gate: tpu7x-512, chunk=${CHUNK}, all levers ON ==="
sudo docker run --rm \
  -v /mnt/disks/scratch/maxtext-1410-upstream/src/maxtext:/deps/src/maxtext:ro \
  -e LIBTPU_INIT_ARGS="$BASEFLAGS" \
  -e JAX_PLATFORMS=cpu -e ENABLE_PJRT_COMPATIBILITY=true \
  "$IMAGE" \
  python3 -m maxtext.trainers.pre_train.train_compile maxtext/configs/base.yml \
  $MODEL_ARGS $COMPILE
