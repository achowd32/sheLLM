#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

input="inputs_outputs/data_1.txt"
output="inputs_outputs/data_2.txt"

if diff <(cat "$input" | LC_ALL=C tr -cd '\0-\177') <(cat "$output"); then
    echo "$0 success: texts are identical"
    exit 0
else
    echo "$0 failure: texts are not identical"
    exit 1
fi
