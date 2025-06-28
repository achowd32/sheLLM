#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

input="inputs_outputs/data_2.txt"
train="inputs_outputs/data_3_train.txt"
val="inputs_outputs/data_3_val.txt"

length=$(wc -c < "$input" | grep -oE "[0-9]+")
split=$((length * 9 / 10))

if diff <(head -c "$split" "$input") <(cat "$train"); then
    echo "$0 (training data) success: texts are identical"
else
    echo "$0 (training data) failure: texts are not identical"
    exit 1
fi
