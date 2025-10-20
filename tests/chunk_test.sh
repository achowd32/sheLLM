#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

input="inputs/chunk_1.txt"
output="outputs/chunk_1.txt"

if diff <(cat "$input" | ../core/chunk.js "$BATCH_SIZE" "$BLOCK_SIZE") <(cat "$output"); then
    echo "$0 success: texts are identical"
    exit 0
else
    echo "$0 failure: texts are not identical"
    exit 1
fi
