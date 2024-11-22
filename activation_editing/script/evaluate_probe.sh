#!/bin/bash

python evaluate_probe.py \
	--base_model="meta-llama/Meta-Llama-3-8B-Instruct" \
	--guidance_modules="./logs/ckpt/truthfulqa/mc2/householder/llama3_8b_instruct_l4_r5e4/guidance_modules" \
	--eval_dataset_path="./data/truthfulqa/mc2/llama3_8b/eval" \
	--batch_size=32 \
	--num_workers=8 \
	--top_k=5
