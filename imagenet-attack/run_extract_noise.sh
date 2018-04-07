#!/bin/bash
PREFIX="latereg-l2tv"

env -i CUDA_VISIBLE_DEVICES="" python attack.py \
    --config_file config_attack/${PREFIX}.json \
    --noise_restore_checkpoint="$1" \
    --just_apply_noise True \
    --apply_folder="/data/experimental_images/mug/val/cropped" 

#env -i CUDA_VISIBLE_DEVICES="" python classify.py "/data/experimental_images/mug/val/cropped"
