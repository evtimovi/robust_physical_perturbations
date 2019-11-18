#!/bin/bash
export SIGN_PREFIX="noinversemask_second_trial_run"
export SIGN_MASK="octagon.png"
export EPOCH=299

python apply_noise_to_bigger_image.py \
    --big_image ./misc/uw17-cropped.png \
    --model_path ./optimization_output/${SIGN_PREFIX}/model/${SIGN_PREFIX}-${EPOCH} \
    --output_path ./misc/uw17-octagon-noise-npadd.png \
    --attack_mask ./masks/octagon.png

