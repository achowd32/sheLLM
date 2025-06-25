#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
BLUE='\033[1;34m'
RESET='\033[0m'
set +o allexport

./../core/tokenize.sh "100" "inputs_outputs/token_input.txt"
