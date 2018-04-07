#!/bin/bash
INPUT_DIR=$1

env CUDA_VISIBLE_DEVICES="" python classify.py \
      --input_dir="${INPUT_DIR}" \
      --output_file="${INPUT_DIR}/classifications.txt" \
      --config_file="config_attack/classify.json"
