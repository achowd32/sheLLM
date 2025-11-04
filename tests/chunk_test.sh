#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.sh"
set +o allexport

for i in $(seq 4); do
    input="inputs/chunk_${i}.txt"
    output="outputs/chunk_${i}.txt"

    if diff <(cat "$input" | ../core/chunk.js "$BATCH_SIZE" "$BLOCK_SIZE") <(cat "$output"); then
        echo "$0 success on test ${i}: texts are identical"
    else
        echo "$0 failure on test ${i}: texts are not identical"
        exit 1
    fi
done

exit 0
