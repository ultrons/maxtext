#!/bin/bash
# Clean reshard on the 8x8x8 itself: restore ckpt0424-fsdp (fsdp=16/ep=1) into our
# fsdp=128/ep=8 mesh, take ONE step at learning_rate=0 (so params are unchanged), save.
# Output: gs://sivaibhav-exp/ckpt-reshard/ds671b-fsdp128-ep8/checkpoints/0/items
set -e
SUF="${1:-reshard1}"
SHIP="quantization=fp8_full use_qwix_quantization=true weight_quantization_calibration_method=fixed,-1,1 \
moe_fp8_cv_weight_ag=true moe_fp8_cv_weight_ag_tags=false moe_ring_cotangent_ag=true \
moe_x_sorted=device bwd_quantization_dtype=e4m3 \
moe_bwd_inkernel_quant=true moe_bwd_inkernel_quant_dlhs=true moe_fold_wo_scale_in_gather=true \
ragged_buffer_factor=-1 num_moe_token_chunks=1"
DATA="dataset_type=tfds dataset_path=gs://mlperf-6-submission-us-central1/tfds-reshard \
dataset_name=c4/en:3.0.5 eval_dataset_name=c4/en:3.0.5 \
train_data_columns=[ids] eval_data_columns=[ids] tokenize_train_data=False tokenize_eval_data=False \
tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken tokenizer_type=tiktoken \
data_shuffle_seed=1234 use_random_routing=false eval_interval=-1"
CKPT="enable_checkpointing=True async_checkpointing=false checkpoint_period=1 \
checkpoint_storage_concurrent_gb=1024 \
load_parameters_path=gs://mlperf-6-submission-us-central1/ckpt0424-fsdp/0/items \
base_output_directory=gs://sivaibhav-exp/ckpt-reshard run_name=ds671b-fsdp128-ep8"
MODEL="mtp_num_layers=1 mtp_loss_scaling_factor=0.1 load_balance_loss_weight=0"
# learning_rate=0 => the one step applies a zero update, so the saved params are the restored params.
IMGTAG=fixes ENVX="MOE_UNSORT_BWD_MASK=1" bash xpk_variant_poison.sh "$SUF" \
  "$SHIP $DATA $CKPT $MODEL learning_rate=0.0 steps=1 profiler=''"
