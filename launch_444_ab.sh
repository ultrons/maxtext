#!/bin/bash
# 4x4x4 (64-chip) A/B for the a2a-dispatch project, Phase A (L1 baseline).
# MODE=ring : 14.10-record config rescaled to FSDP=32 x EP=4 (control)
# MODE=a2a  : use_ring_of_experts=false variant (existing unfused ragged_all_to_all path)
# Same image as the record (composeall-5c01ac873) — no code changes, same-image A/B.
# Submits through Kueue micro-queue (jakeriley cluster, tpu-4x4x4-b preferred / -c overflow).
# Usage: MODE=<ring|a2a> RUN_TAG=<tag> [LAYERS=27] bash launch_444_ab.sh
set -e
set -o pipefail

: "${MODE:?set MODE=ring or a2a}"
: "${RUN_TAG:?set RUN_TAG}"
LAYERS="${LAYERS:-27}"

export PROJECT_ID="tpu-prod-env-automated"
export CLUSTER_NAME="jakeriley-v7x-do-not-delete"
export ZONE="us-central1-c"
export KCTX="gke_tpu-prod-env-automated_us-central1_jakeriley-v7x-do-not-delete"
export BASE_OUTPUT_DIR="gs://sivaibhav-exp/maxtext"
export ARTIFACT_DIR="gs://sivaibhav-exp/maxtext/${RUN_TAG}"
export WORKLOAD_IMAGE="gcr.io/tpu-prod-env-automated/integrate_v2:composeall-5c01ac873"
export WORKLOAD_NAME="${RUN_TAG}"

XLA_FLAGS=" $(grep -vE '^\s*(#|$)' flags_record_1410.txt | sed 's/[[:space:]]*$//' | tr '\n' ' ') "

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
steps=20 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME} \
profiler=xplane \
skip_first_n_steps_for_profiler=5 \
profiler_steps=3"

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
MAXTEXT_ARGS="$COMMON_ARGS $MODE_ARGS"

MANIFEST="/tmp/${WORKLOAD_NAME}_jobset.yaml"

# xpk dry-run emits the JobSet skeleton; capture the YAML (starts at 'apiVersion').
xpk workload create \
  --dry-run \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --max-restarts=0 \
  --device-type=tpu7x-4x4x4 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --no-use-parallel-containers \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log ; \
gcloud storage cp --no-user-output-enabled train.log ${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID}.log" \
  | awk '/^apiVersion:/,/^\[XPK\]/' | sed '/^\[XPK\]/d' > "$MANIFEST"

test -s "$MANIFEST" || { echo "manifest capture failed"; exit 1; }

# Patch for the micro-queue route (TPU_CAPACITY.md):
#  1. multislice-queue -> micro-queue + micro-low priority
#  2. suspend: true (Kueue admits/unsuspends)
#  3. DROP the stale placement-policy-name selector (known xpk trap on this cluster)
sed -i 's/kueue.x-k8s.io\/queue-name: multislice-queue.*/kueue.x-k8s.io\/queue-name: micro-queue\n    kueue.x-k8s.io\/priority-class: micro-low/' "$MANIFEST"
sed -i '/cloud.google.com\/placement-policy-name/d' "$MANIFEST"
grep -q "^spec:" "$MANIFEST" && sed -i '0,/^spec:/s//spec:\n  suspend: true/' "$MANIFEST"

echo "=== patched manifest: $MANIFEST ==="
grep -nE "suspend|kueue|placement|parallelism|google.com/tpu|topology" "$MANIFEST"

if [ -n "${DRY_RUN:-}" ]; then echo "(DRY_RUN set — not applying)"; exit 0; fi
kubectl --context "$KCTX" apply -f "$MANIFEST"
echo "submitted ${WORKLOAD_NAME}; watch: kubectl --context $KCTX get jobsets,workloads"
