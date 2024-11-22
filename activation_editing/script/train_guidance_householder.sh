#!/bin/bash -e

SAVE_DIR="./logs/ckpt/truthfulqa/mc2/householder/llama3_8b_instruct_l4_r5e4"

CUDA_VISIBLE_DEVICES=0 python train_guidance.py \
	--base_model="meta-llama/Meta-Llama-3-8B-Instruct" \
	--train_dataset_path="./data/truthfulqa/mc2/llama3_8b/train" \
	--eval_dataset_path="./data/truthfulqa/mc2/llama3_8b/eval" \
	--save_dir="${SAVE_DIR}" \
	--keep_in_memory=False \
	--target_layers="all" \
	--guidance_module_type="householder" \
	--num_guidance_module_layers=4 \
	--learning_rate 5e-4 \
	--warmup_ratio 0.05 \
	--num_train_epochs 5 \
	--evaluation_strategy "epoch" \
	--prediction_loss_only False \
	--report_to "none" \
	--save_strategy "epoch" \
	--load_best_model_at_end True \
	--metric_for_best_model="lr_acc" \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--logging_steps 200 \
	--log_level="info" \
	--log_level_replica="info" \
	--output_dir="${SAVE_DIR}/trainer_outputs" \
	--save_total_limit=5 \
	--skip_memory_metrics False \
	--dataloader_num_workers=8 \
	--dataloader_pin_memory False
