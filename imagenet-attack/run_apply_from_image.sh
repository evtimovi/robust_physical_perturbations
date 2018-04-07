#!/bin/bash
PREFIX="latereg-l2tv"
EPOCH=29900
python apply_from_image.py \
    output_noisegen/${PREFIX}/noise-epoch-${EPOCH}.png \
    /data/experimental_images/mug/val/cropped/ \
    masks/mask-mug-299.png
