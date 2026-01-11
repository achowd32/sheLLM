#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.sh"
set +o allexport

for i in $(seq 4); do
    input="../tests/inputs/tok_${i}.txt"
    output="../tests/outputs/tok_${i}.txt"
    if diff <(../core/tokenize.sh "8" "$input" | awk '{$1=$1;print}') <(cat "$output"); then
        echo "$0 success on test #${i}: texts are identical"
    else
        echo "$0 failure on test #${i}: texts are not identical"
        exit 1
    fi
done

exit 0
