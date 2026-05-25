#!/bin/bash
GAN_TYPE=$1
DATASET=$2

if [ -z "$GAN_TYPE" ] || [ -z "$DATASET" ]; then
    echo "Usage: bash run.sh <gan_type> <dataset>"
    echo "Example: bash run.sh ctgan ml-100k"
    echo "Example: bash run.sh baseline ml-100k"
    exit 1
fi

echo ""
echo "### STEP 1: Creating $DATASET CSV for GAN Training ###"
echo ""
python3 "utils.py" --dataset "$DATASET"

if [ "$GAN_TYPE" == "baseline" ]; then
    echo ""
    echo "### Baseline mode: Skipping preprocessing, GAN, and model training ###"
    echo ""
else
    if [ "$GAN_TYPE" == "tabfairgan" ] || [ "$GAN_TYPE" == "decaf" ] || [ "$GAN_TYPE" == "cfgan" ]; then
        echo ""
        echo "### STEP 2: Skipping Fairness Preprocessing (DECAF/CFGAN handle fairness internally) ###"
        echo ""
    else
        echo ""
        echo "### STEP 2: Fairness Preprocessing ###"
        echo ""
        python3 "preprocess_fair.py" --dataset "$DATASET"
    fi
    
    echo ""
    echo "### STEP 3: GAN Training ###"
    echo ""
    python3 "$GAN_TYPE/main.py" --dataset "$DATASET"
    
    echo ""
    echo "### STEP 4: Model Training on Augmented Data ###"
    echo ""
    python3 "pretrained/$DATASET.py" --gan "$GAN_TYPE" --dataset "$DATASET"
fi

echo ""
echo "### STEP 5: Evaluation ###"
echo ""
python3 "eval.py" --gan "$GAN_TYPE" --dataset "$DATASET"