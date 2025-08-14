#!/bin/bash

cd "$(dirname $0)"

mkdir logs/

set -o allexport
source arch/hyperparameters.py
BLUE='\033[1;34m'
RESET='\033[0m'
set +o allexport

# ------------------- START PIPELINE ------------------- 

# initialize variables
data_file="data.txt"
train_file="train.txt"
val_file="val.txt"
model_file="model"
sample_file="sample.txt"
eval_type="lang" # can be "lang", "bert", or "posp"

# process data
data/data.sh "$data_file" "$train_file" "$val_file"

# run core tokenization and training pipeline
core/core.sh "$train_file" "$val_file" "$model_file"

# perform evaluations
eval/eval.sh "$val_file" "$model_file" "$sample_file" "$eval_type"
