#!/bin/bash

cd "$(dirname $0)"
mkdir logs/

# create environment variables, accessible in all subshells
set -o allexport

# set all hyperparameters
eval $(awk 'NR <= 10 { print "export", $2 "=" $4 }' arch/hyperparameters.js)

# set color ANSI codes 
BLUE='\033[1;34m'; RESET='\033[0m'

set +o allexport

# ------------------- START PIPELINE ------------------- 

# process data
data/data.sh

# run core tokenization and training pipeline
core/core.sh

# perform evaluations
eval/eval.sh
