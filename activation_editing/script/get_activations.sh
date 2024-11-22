#!/bin/bash

python get_activations.py \
    --model="meta-llama/Meta-Llama-3-8B-Instruct" \
    --save_dir="./data/truthfulqa/mc2/llama3_8b/" \
    --dataset="truthful_qa" \
    --task="multiple_choice" \
    --sub_task="mc2" \
    --data_splits_indices_path="./data/truthfulqa/ids_splits.json" \
    --target_split="train,eval"
