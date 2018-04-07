#!/bin/bash
python classify_yadav.py \
    --srcimgs ../usstops-for-test-set/resized32 \
    --weights ./models/gtsrb_usstop/model_best_test \
    --img_rows 32 \
    --img_cols 32 \
    --nb_classes 43
