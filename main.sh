#!/bin/bash

cd "$(dirname "$0")"
mkdir logs/
mkdir outputs/

# create environment variables, accessible in all subshells
set -o allexport

# set all hyperparameters
eval $(awk 'NR <= 10 { print "export", $2 "=" $4 }' arch/hyperparameters.js)

# set color ANSI codes 
BLUE='\033[1;34m'; RESET='\033[0m'

set +o allexport

# ------------------- START PIPELINE ------------------- 

# initialize variables
data_file="data.txt" # name of the file to which we write all data
train_file="train.txt" # name of the file to which we write training data
val_file="val.txt" # name of the file to which we write validation data
model_dir="model" # name of the directory to which we save our trained model
output_toks=500 # number of tokens per generated output
output_prompt="" # prompt used for output generation
num_outputs=3 # number of outputs to generate

# process data
data/data.sh "$data_file" "$train_file" "$val_file"

# run core tokenization and training pipeline
core/core.sh "$train_file" "$val_file" "$model_dir"

# prompt model and generate outputs
generate/generate.sh "$model_dir" "$output_toks" "$output_prompt" "$num_outputs"

# perform evaluations
prompt_len=$(echo -n "$output_prompt" | wc -c | tr -d ' ')
eval/eval.sh "$val_file" "$((prompt_len + output_toks))"
