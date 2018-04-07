#!/bin/bash

export SIGN_PREFIX="usstop-l1rectangles-less-256"
export SIGN_MASK="mask_l1rectangles-less_256.png"
export SIGN_TARGET=5

mkdir optimization_output/${SIGN_PREFIX}
mkdir optimization_output/${SIGN_PREFIX}/noisy_images
mkdir optimization_output/${SIGN_PREFIX}/model

python gennoise_yadav_model.py \
    --input_rows 256 \
    --input_cols 256 \
    --resize_method convresize \
    --true_class 14 \
    --save_all_noisy_images False \
    --device="/cpu:0" \
    --min_rate_to_save 0.5 \
    --nb_classes 43 \
    --img_rows 32 \
    --img_cols 32 \
    --optimization_rate 0.25 \
    --regloss l2 \
    --optimization_loss justcrossentropy \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 0.0001 \
    --model_path="./models/gtsrb_usstop/model_best_test" \
    --attack_epochs 1000 \
    --attack_lambda 0.001 \
    --attack_srcdir="./victim-set" \
    --attack_mask="./masks/${SIGN_MASK}" \
    --checkpoint ${SIGN_PREFIX}\
    --target_class ${SIGN_TARGET} \
    --printability_optimization False \
    --clipping True \
    --noise_clip_max 20.0 \
    --noise_clip_min -20.0 \
    --noisy_input_clip_max 0.5 \
    --noisy_input_clip_min -0.5 | tee ./optimization_output/${SIGN_PREFIX}/optimization_printout_${SIGN_PREFIX}.txt

