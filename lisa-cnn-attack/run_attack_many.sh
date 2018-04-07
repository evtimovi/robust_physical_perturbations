#!/bin/bash
export SIGN_PREFIX="octagon"
export SIGN_SETPOINT=0.37
export SIGN_MASK="octagon.png"
export SIGN_TARGET=12

mkdir optimization_output/${SIGN_PREFIX}
mkdir optimization_output/${SIGN_PREFIX}/noisy_images
mkdir optimization_output/${SIGN_PREFIX}/model

python gennoise_many_images.py  \
    --tf_seed 12345 \
    --inverse_mask False \
    --initial_value_for_noise="" \
    --fullres_input False \
    --optimization_rate 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --model_path="./models/all_r_ivan.ckpt" \
    --attack_epochs 300 \
    --save_frequency 10 \
    --attack_srcdir="victim-set" \
    --attack_mask="./masks/${SIGN_MASK}" \
    --checkpoint ${SIGN_PREFIX}\
    --target_class ${SIGN_TARGET} \
    --inverse_mask_setpoint ${SIGN_SETPOINT} \
    --printability_optimization False \
    --printability_tuples="npstriplets.txt" \
    --clipping True \
    --noise_clip_max 20.0 \
    --noise_clip_min -20.0 \
    --noisy_input_clip_max 1.0 \
    --noisy_input_clip_min 0.0 | tee ./optimization_output/${SIGN_PREFIX}/optimization_printout_${SIGN_PREFIX}.txt

