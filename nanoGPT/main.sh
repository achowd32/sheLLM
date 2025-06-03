#!/bin/bash

cd "$(dirname $0)"

mkdir data/ model/

set -o allexport
source arch/hyperparameters.py
set +o allexport

mkfifo /tmp/vs_pipe
cd init; ./init.sh
export VOCAB_SIZE="$(cat /tmp/vs_pipe)"
rm /tmp/vs_pipe

cd ../core; ./core.sh

cd ../eval; ./eval.sh

