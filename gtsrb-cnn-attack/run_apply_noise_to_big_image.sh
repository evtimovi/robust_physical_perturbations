#!/bin/bash
export SIGN_PREFIX="usstop-l1rectangles-less-256"
export SIGN_MASK="mask_l1rectangles-less_256.png"
export EPOCH=999
export RESIZE_METHOD="area"

python apply_noise_to_bigger_image.py \
    --downsize_first True \
    --img_cols 256 \
    --img_rows 256 \
    --nb_channels 3 \
    --big_image ./uw17-square.jpg \
    --output_path ./uw17-square-${SIGN_PREFIX}-epoch-${EPOCH}-${RESIZE_METHOD}-downsizefirst.png \
    --model_path ./optimization_output/${SIGN_PREFIX}/model/${SIGN_PREFIX}-${EPOCH} \
    --attack_mask ./masks/${SIGN_MASK} \
    --device="/cpu" \
    --resize_method ${RESIZE_METHOD} \
    --resize_rows 2484 \
    --resize_cols 2484 \
    --resize_noise_only False 
