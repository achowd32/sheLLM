#!/bin/bash

cd "$(dirname $0)"

mkdir data/ model/ logs/

set -o allexport
source arch/hyperparameters.py
BLUE='\033[1;34m'
RESET='\033[0m'
set +o allexport

mkfifo /tmp/vs_pipe
cd init; ./init.sh
export VOCAB_SIZE="$(cat /tmp/vs_pipe)"
rm /tmp/vs_pipe

cd ../core; ./core.sh

cd ../eval; ./eval.sh
