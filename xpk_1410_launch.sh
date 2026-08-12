#!/bin/bash
# Launch the 14.10 re-run on bodaborg via xpk (correct 4x8x8 provisioning structure) +
# inject the siv-preempt (2000) Kueue priority so it preempts the gu-ubench harness (1000).
# xpk --priority maxes at very-high(1000) which can't preempt, so we dry-run + patch the label.
set -e
set -o pipefail
cd /mnt/disks/scratch/maxtext-1410-upstream
source ~/xdb/.xprof/bin/activate 2>/dev/null

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
steps=30 base_output_directory=gs://sivaibhav-exp/1410-upstream run_name=siv-1410-4x8x8 \
profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=3"

CMD="export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && python3 -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} 2>&1 | tee train.log"

# 1) xpk dry-run -> capture generated manifest
xpk workload create \
  --cluster=bodaborg-tpu7x-nap --project=cloud-tpu-shared-capacity --zone=us-central1-c \
  --device-type=tpu7x-4x8x8 --num-slices=1 \
  --reservation=cloudtpu-20260710003900-159478293 \
  --priority=very-high --max-restarts=3 --no-use-parallel-containers \
  --docker-image="gcr.io/tpu-vm-gke-testing/integrate_v2:1410-upstream-ac9498b64" \
  --workload=siv-1410-4x8x8 \
  --command="$CMD" \
  --dry-run > /tmp/xpk_full.txt 2>&1

# 2) extract the JobSet YAML (from 'apiVersion:' to just before the first [XPK] log line after it)
awk '/^apiVersion: jobset/{p=1} p&&/^\[XPK\]/{exit} p{print}' /tmp/xpk_full.txt > /tmp/xpk_manifest.yaml
echo "extracted manifest lines: $(wc -l < /tmp/xpk_manifest.yaml)"

# 3) inject the siv-preempt (2000) Kueue WorkloadPriorityClass label under metadata.labels
python3 - <<'PY'
p="/tmp/xpk_manifest.yaml"; L=open(p).read().split("\n"); out=[]
done=False
for i,l in enumerate(L):
    out.append(l)
    if (not done) and l.strip().startswith("kueue.x-k8s.io/queue-name:"):
        indent=l[:len(l)-len(l.lstrip())]
        out.append(f"{indent}kueue.x-k8s.io/priority-class: siv-preempt")
        done=True
open(p,"w").write("\n".join(out))
print("injected siv-preempt priority-class label:", done)
PY
echo "=== priority + queue labels in final manifest ==="
grep -nE 'queue-name|priority-class|priorityClassName' /tmp/xpk_manifest.yaml
