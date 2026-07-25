#!/bin/bash
# AOT memory/compile gate for the 4x4x4 (64-chip / 128-core) a2a-vs-ring A/B.
# MODE=ring : the 14.10 record config (recovered from ds-v3-composeon-0704 train log)
#             rescaled to FSDP=32 x EP=4, base_num_decoder_layers=${LAYERS}.
# MODE=a2a  : same, minus the ring-only levers (see perf-drills gather/a2a_fused/A1_FLAG_AUDIT.md):
#             use_ring_of_experts=false, unchunked, stock autodiff bwd.
# Same image as the record: integrate_v2:composeall-5c01ac873 (no code changes needed).
# Usage: MODE=<ring|a2a> [LAYERS=27] bash aot_444_a2a.sh
set -e
set -o pipefail

: "${MODE:?set MODE=ring or a2a}"
LAYERS="${LAYERS:-27}"
IMAGE=gcr.io/tpu-prod-env-automated/integrate_v2:composeall-5c01ac873
BASEFLAGS=$(grep -vE '^\s*(#|$)' flags_record_1410.txt | tr '\n' ' ')

COMMON_ARGS="\
model_name=deepseek3-671b \
base_num_decoder_layers=${LAYERS} \
override_model_config=True \
per_device_batch_size=4.0 \
max_target_length=4096 \
ici_fsdp_parallelism=32 \
ici_expert_parallelism=4 \
ici_data_parallelism=1 \
ici_fsdp_transpose_parallelism=1 \
dcn_data_parallelism=-1 \
dcn_pipeline_parallelism=1 \
ici_pipeline_parallelism=1 \
shard_exp_on_fsdp=False \
use_iota_embed=True \
tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
opt_type=adamw \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
dtype=bfloat16 \
sa_use_fused_bwd_kernel=True \
megablox=True \
sparse_matmul=True \
use_tokamax_gmm=True \
use_gmm_v2=True \
use_tokamax_splash=True \
use_max_logit_estimate=-1 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False \
remat_policy=custom \
allow_split_physical_axes=False \
decoder_layer_input=device \
enable_tpu_profiling_options=True \
async_checkpointing=False \
enable_checkpointing=False \
attention=flash \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_kv_dq=2048 \
sa_block_q_dq=2048 \
use_random_routing=True \
moe_routing_key_as_input=True \
use_custom_sort_vjp=True \
use_ragged_sort=True \
merge_gating_gmm=False \
ragged_buffer_factor=-1 \
optimizer_memory_host_offload=False \
wi_tile_fwd_batch_seq=256 \
wi_tile_fwd_embed_dim=7168 \
wi_tile_fwd_mlp_dim=1024 \
wi_tile_dlhs_batch_seq=256 \
wi_tile_dlhs_embed_dim=3584 \
wi_tile_dlhs_mlp_dim=2048 \
wi_tile_drhs_batch_seq=512 \
wi_tile_drhs_embed_dim=1792 \
wi_tile_drhs_mlp_dim=2048 \
wo_tile_fwd_batch_seq=512 \
wo_tile_fwd_embed_dim=3584 \
wo_tile_fwd_mlp_dim=2048 \
wo_tile_dlhs_batch_seq=512 \
wo_tile_dlhs_embed_dim=1792 \
wo_tile_dlhs_mlp_dim=2048 \
wo_tile_drhs_batch_seq=512 \
wo_tile_drhs_embed_dim=1792 \
wo_tile_drhs_mlp_dim=2048 \
skip_jax_distributed_system=True \
steps=20"

RING_ARGS="\
use_ring_of_experts=True \
moe_n_chunks=2 \
decouple_combine_rs_chunks=4 \
moe_chunked_combine_in_remat=True \
moe_handwritten_bwd=True \
moe_weight_ag_scheduling_group=True \
moe_splash_host_offload=True \
moe_direct_rs=True \
moe_direct_token_ag=True \
moe_direct_combine_ag=True \
qk_diag_skip=True \
qk_diag_grid=4"

A2A_ARGS="\
use_ring_of_experts=False \
moe_n_chunks=1 \
decouple_combine_rs_chunks=0 \
moe_chunked_combine_in_remat=False \
moe_handwritten_bwd=False \
moe_weight_ag_scheduling_group=False \
moe_splash_host_offload=False \
moe_direct_rs=False \
moe_direct_token_ag=False \
moe_direct_combine_ag=False \
qk_diag_skip=True \
qk_diag_grid=4"

if [ "$MODE" = "ring" ]; then MODE_ARGS="$RING_ARGS"; else MODE_ARGS="$A2A_ARGS"; fi

COMPILE="compile_topology=tpu7x-128 compile_topology_num_slices=1 base_output_directory=/tmp/aot_out run_name=aot444_${MODE}_l${LAYERS}"

sudo docker run --rm \
  -v "$PWD":/host_maxtext \
  -e LIBTPU_INIT_ARGS="$BASEFLAGS" \
  -e JAX_PLATFORMS=cpu -e ENABLE_PJRT_COMPATIBILITY=true \
  $IMAGE \
  python3 -m maxtext.trainers.pre_train.train_compile maxtext/configs/base.yml \
  $COMMON_ARGS $MODE_ARGS $COMPILE
