#!/bin/bash

cd "$(dirname $0)"

mkdir logs/

set -o allexport
source arch/hyperparameters.py
BLUE='\033[1;34m'
RESET='\033[0m'
set +o allexport

cd data; ./data.sh

cd ../core; ./core.sh

cd ../eval; ./eval.sh
