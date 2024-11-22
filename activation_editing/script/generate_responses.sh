#!/bin/bash -e

python generate_responses.py \
	--base_model="meta-llama/Meta-Llama-3-8B-Instruct" \
	--guidance_modules="./logs/ckpt/truthfulqa/mc2/householder/llama3_8b_instruct_l4_r5e4/guidance_modules" \
	--data_splits_indices_path="./data/truthfulqa/ids_splits.json" \
	--target_splits="test" \
	--output_dir="./logs/gen_outputs/truthfulqa/mc2/llama3_8b" \
	--dataset="truthful_qa" \
	--task="multiple_choice" \
	--sub_task="mc2" \
	--batch_size=8 \
	--max_new_tokens=100
