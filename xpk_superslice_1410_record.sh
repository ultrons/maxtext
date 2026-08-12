#!/bin/bash
# Launch the 14.10 RECORD on bodaborg-super-tpu7x-y6k at 4x8x8 (256 chips) via xpk super-slicing.
# Recipe: infra/superslice-y6k-4x8x8-recipe.md — SUB_SLICING_ENABLED=true + --priority=very-high (>competitors)
# kills the preemption churn that caused the VF-session/TearDownMesh crash.
set -e
set -o pipefail
cd /mnt/disks/scratch/maxtext-1410-upstream

CLUSTER=bodaborg-super-tpu7x-y6k
PROJECT=cloud-tpu-multipod-dev
ZONE=us-central1-c
RES=ghostfish-y6kzjgh4shlno
IMG=gcr.io/tpu-vm-gke-testing/integrate_v2:1410-up2-d59a6a618
NAME=siv-1410-up2

XLA_FLAGS="$(grep -vE '^\s*(#|$)' flags_record_1410.txt | sed 's/[[:space:]]*$//' | tr '\n' ' ')"
CHUNK="${CHUNK:-2}"

MAXTEXT_ARGS="model_name=deepseek3-671b per_device_batch_size=4.0 max_target_length=4096 \
ici_fsdp_parallelism=128 ici_expert_parallelism=4 ici_data_parallelism=1 ici_fsdp_transpose_parallelism=1 \
dcn_data_parallelism=-1 dcn_pipeline_parallelism=1 ici_pipeline_parallelism=1 \
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
optimizer_memory_host_offload=True \
moe_weight_ag_scheduling_group=True moe_routing_key_as_input=True moe_handwritten_bwd=True \
moe_splash_host_offload=True num_moe_token_chunks=${CHUNK} decouple_combine_rs_chunks=4 \
moe_chunked_combine_in_remat=True moe_direct_rs=True moe_direct_token_ag=True moe_direct_combine_ag=True \
qk_diag_skip=True qk_diag_grid=4 \
steps=30 base_output_directory=gs://sivaibhav-exp/1410-up2 run_name=${NAME} \
profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=3"

CMD="export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee /tmp/train.log"

echo "=== launching $NAME : 4x8x8 (256 chips) super-slicing, priority very-high, chunk=$CHUNK ==="
SUB_SLICING_ENABLED=true xpk workload create \
  --cluster=$CLUSTER --project=$PROJECT --zone=$ZONE \
  --device-type=tpu7x-4x8x8 --num-slices=1 \
  --reservation=$RES \
  --priority=very-high --max-restarts=5 \
  --docker-image="$IMG" \
  --workload=$NAME \
  --command="$CMD"
