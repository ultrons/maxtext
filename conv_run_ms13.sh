#!/bin/bash
# DSv3 fp8 convergence run to target_eval_loss=3.60.
# THEIRS: LR schedule, dataset, tokenizer, MTP, grouped routing, eval cadence, early-stop target.
# OURS:   pdbs=1, fsdp=128/ep=8 on 8x8x8, fp8 ship stack, rbf=-1, chunks=2.
set -e
SUF="${1:-ms13}"
SHIP="quantization=fp8_full use_qwix_quantization=true weight_quantization_calibration_method=fixed,-1,1 \
moe_fp8_cv_weight_ag=true moe_fp8_cv_weight_ag_tags=false moe_ring_cotangent_ag=true \
moe_x_sorted=device bwd_quantization_dtype=e4m3 \
moe_bwd_inkernel_quant=true moe_bwd_inkernel_quant_dlhs=true moe_fold_wo_scale_in_gather=true \
ragged_buffer_factor=-1 num_moe_token_chunks=2 per_device_batch_size=1.0 dcn_data_parallelism=13"
DATA="dataset_type=tfds dataset_path=gs://mlperf-6-submission-us-central1/tfds-reshard \
dataset_name=c4/en:3.0.5 eval_dataset_name=c4/en:3.0.5 \
train_data_columns=[ids] eval_data_columns=[ids] tokenize_train_data=False tokenize_eval_data=False \
tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken tokenizer_type=tiktoken \
data_shuffle_seed=1234 use_random_routing=false"
CKPT="enable_checkpointing=True async_checkpointing=false checkpoint_period=100 \
checkpoint_storage_concurrent_gb=1024 base_output_directory=gs://sivaibhav-exp/dsv3-fp8-conv \
load_parameters_path=gs://mlperf-6-submission-us-central1/ckpt0424-fsdp/0/items"
MODEL="mtp_num_layers=1 mtp_loss_scaling_factor=0.1 \
n_routing_groups=8 topk_routing_group=4 load_balance_loss_weight=0"
LR="learning_rate=0.000024 lr_schedule_type=cosine learning_rate_schedule_steps=12000 \
warmup_steps_fraction=0.00033333 learning_rate_final_fraction=0.000417"
EVAL="eval_interval=1 eval_steps=1 target_eval_loss=3.60"
SAFE="abort_on_nan_loss=true abort_on_inf_loss=true"
IMGTAG=fixes ENVX="MOE_UNSORT_BWD_MASK=1" bash xpk_variant_ms13.sh "$SUF" \
  "$SHIP $DATA $CKPT $MODEL $LR $EVAL $SAFE steps=12000"
