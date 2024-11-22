#!/bin/bash


python cli_demo.py \
	--base_model="meta-llama/Meta-Llama-3-8B-Instruct" \
	--guidance_module="./logs/ckpt/truthfulqa/mc2/householder/llama3_8b_instruct_l4_r5e4/guidance_modules" \
	--max_new_tokens=100
