#!/bin/bash
set -e
if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh [ctgan|fairgan] [ml-100k|ml-200k]"
    exit 1
fi
GAN_TYPE=$1
DATASET=$2
if [ "$GAN_TYPE" != "ctgan" ] && [ "$GAN_TYPE" != "fairgan" ]; then
    echo "Invalid GAN type: $GAN_TYPE"
    exit 1
fi
if [ "$DATASET" != "ml-100k" ] && [ "$DATASET" != "ml-200k" ]; then
    echo "Invalid dataset: $DATASET"
    exit 1
fi

python3 "utils.py" --dataset "$DATASET"
python3 "$GAN_TYPE/main.py" --dataset "$DATASET"
python3 "pretrained/$DATASET.py" --gan "$GAN_TYPE" --dataset "$DATASET"
python3 "eval.py" --gan "$GAN_TYPE" --dataset "$DATASET"