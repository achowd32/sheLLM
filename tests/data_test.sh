#!/bin/bash

cd "$(dirname "$0")"

for i in $(seq 3); do
    input="../tests/inputs/data_${i}.txt"
    train_output="../tests/outputs/data_${i}_train.txt"
    val_output="../tests/outputs/data_${i}_val.txt"

    tmp_train="/tmp/temp_train.txt"
    tmp_val="/tmp/temp_val.txt"

    ../data/data.sh "$input" "$tmp_train" "$tmp_val" >/dev/null

    if diff "$tmp_train" "$train_output"; then
        echo "$0 (training data) success on test #${i}: texts are identical"
    else
        echo "$0 (training data) failure on test #${i}: texts are not identical"
        rm "$tmp_train" "$tmp_val"
        exit 1
    fi

    if diff "$tmp_val" "$val_output"; then
        echo "$0 (validation data) success on test #${i}: texts are identical"
    else
        echo "$0 (validation data) failure on test #${i}: texts are not identical"
        rm "$tmp_train" "$tmp_val"
        exit 1
    fi
done

rm "$tmp_train" "$tmp_val"
exit 0
