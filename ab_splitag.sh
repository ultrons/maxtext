#!/bin/bash
# Same-image A/B for the split-phase (start/done) shared-expert weight all-gather.
# Arm A = stock 6.864 baseline (hoist OFF, split OFF). Arm B = shared_expert_weight_ag_split=true.
# Real c4 + reuse_example_batch=1 (synthetic leaves SC near-idle and removes the contention
# under test). Image 1410-up2-splitag4 carries the VJP fix (identity+reshard; psum_scatter
# over-counted by n and is the likely cause of the hoist's 0.174 lm_loss delta).
set -e
SHIP="quantization=fp8_full use_qwix_quantization=true weight_quantization_calibration_method=fixed,-1,1 \
moe_fp8_cv_weight_ag=true moe_fp8_cv_weight_ag_tags=false moe_ring_cotangent_ag=true \
moe_x_sorted=device bwd_quantization_dtype=e4m3 moe_bwd_inkernel_quant=true \
moe_bwd_inkernel_quant_dlhs=true moe_fold_wo_scale_in_gather=true \
ragged_buffer_factor=-1 num_moe_token_chunks=2 moe_fast_group_topk=true"
DATA="dataset_type=tfds dataset_path=gs://mlperf-6-submission-us-central1/tfds-reshard \
dataset_name=c4/en:3.0.5 eval_dataset_name=c4/en:3.0.5 \
train_data_columns=[ids] eval_data_columns=[ids] tokenize_train_data=False tokenize_eval_data=False \
tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken tokenizer_type=tiktoken \
data_shuffle_seed=1234 use_random_routing=false reuse_example_batch=1"
MODEL="mtp_num_layers=1 mtp_loss_scaling_factor=0.1 \
n_routing_groups=8 topk_routing_group=4 load_balance_loss_weight=0 eval_interval=-1 steps=20"
IMGTAG=splitag4 ENVX="MOE_UNSORT_BWD_MASK=0" bash xpk_variant_poison.sh sag0 "$SHIP $DATA $MODEL"
IMGTAG=splitag4 ENVX="MOE_UNSORT_BWD_MASK=0" bash xpk_variant_poison.sh sag1 "$SHIP $DATA $MODEL shared_expert_weight_ag_split=true"
