#!/bin/bash

CONFIG=$1
GPUS=$2
PY_ARGS=${@:3}

# Make conda available
eval "$(conda shell.bash hook)"
# Activate a conda environment
conda activate shift-tta

python tools/train.py \
    ${CONFIG} \
    ${PY_ARGS}