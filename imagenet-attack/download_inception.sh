#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="clean_checkpoints"

# Download inception v3 checkpoint into base_inception_model subdirectory
cd "${SCRIPT_DIR}/base_inception_model/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz
