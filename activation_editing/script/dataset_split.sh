#!/bin/bash

python dataset_split.py \
  --save_dir="./data/truthfulqa/" \
  --dataset="truthfulqa" \
  --task="multiple_choice" \
  --test_ratio=0.5 \
  --val_ratio=0.1
