#!/bin/bash

cd "$(dirname "$0")"

input="../tests/inputs/data_1.txt"
train_output="../tests/outputs/data_1_train.txt"
val_output="../tests/outputs/data_1_val.txt"

tmp_train="/tmp/temp_train.txt"
tmp_val="/tmp/temp_val.txt"

../data/data.sh "$input" "$tmp_train" "$tmp_val" >/dev/null

if diff "$tmp_train" "$train_output"; then
    echo "$0 (training data) success: texts are identical"
else
    echo "$0 (training data) failure: texts are not identical"
    exit 1
fi

if diff "$tmp_val" "$val_output"; then
    echo "$0 (validation data) success: texts are identical"
    exit 0
else
    echo "$0 (validation data) failure: texts are not identical"
    exit 1
fi

rm "$tmp_train" "$tmp_val"
