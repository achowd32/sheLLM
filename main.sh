#!/bin/bash

cd "$(dirname "$0")"
mkdir logs/

# create environment variables, accessible in all subshells
set -o allexport

# set all hyperparameters
eval $(awk 'NR <= 10 { print "export", $2 "=" $4 }' arch/hyperparameters.js)

# set color ANSI codes 
BLUE='\033[1;34m'; RESET='\033[0m'

set +o allexport

# ------------------- START PIPELINE ------------------- 

# initialize variables
data_file="data.txt"
train_file="train.txt"
val_file="val.txt"
model_dir="model"
sample_file="sample.txt"
num_evals=1

# process data
data/data.sh "$data_file" "$train_file" "$val_file"

# run core tokenization and training pipeline
core/core.sh "$train_file" "$val_file" "$model_dir"

# perform evaluations
eval/eval.sh "$val_file" "$model_dir" "$sample_file" "lang"
