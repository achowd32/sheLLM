#!/bin/bash

cd "$(dirname $0)"

mkdir logs/

# create environment variables, accessible in all subshells
set -o allexport
eval $(awk 'NR <= 10 { print "export", $2 "=" $4 }' arch/hyperparameters.js)
BLUE='\033[1;34m'
RESET='\033[0m'
set +o allexport

# process data
cd data; ./data.sh

# run core tokenization and training pipeline
cd ../core; ./core.sh

# perform evaluations
#cd ../eval; ./eval.sh
