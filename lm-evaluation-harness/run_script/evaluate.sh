#!/bin/bash

CUDA_VISIBLE_DEVICES=0 lm_eval \
	--model="guided" \
	--model_args="pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype=float32,guidance_modules_path=../activation_editing/logs/ckpt/truthfulqa/mc2/householder/llama3_8b_instruct_l4_r5e4/guidance_modules" \
	--tasks="truthfulqa_subset_mc1,truthfulqa_subset_mc2" \
	--device="cuda" \
	--batch_size=4 \
	--output_path="./run_results/llama3/truthfulqa/results.json"
