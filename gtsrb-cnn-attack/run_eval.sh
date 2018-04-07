#!/bin/bash
python eval_yadav.py \
    --nb_classes 43 \
    --weights ./models/german/model_best_test \
    --labels_filename labels.csv \
    --test_dataset ./clean_model/test_data/resized
