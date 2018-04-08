#!/bin/bash
PREFIX="replicate-smallestmask-smallertv"

TARGET_DIR=output_noisegen/${PREFIX}
if [ ! -e $TARGET_DIR  ] 
then
    mkdir $TARGET_DIR
else
    echo "Directory ${TARGET_DIR} already exists, exiting."
    exit 0
fi

env CUDA_VISIBLE_DEVICES="0" python attack.py \
        --config_file="config_attack/${PREFIX}.json" \
        --save_prefix="${PREFIX}" \
        2> errors_${PREFIX}.txt \
        | tee output_noisegen/${PREFIX}/params.txt 

