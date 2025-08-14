#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

sample="This is a sample sentence, which is identical to the other."
reference="This is a sample sentence, which is identical to the other."
pos_score=$(python3 ../eval/pos_eval.py "$sample" "$reference")

if [ "$pos_score" == "1.0000" ]; then
    echo "$0 success: POS-scores correctly calculated"
    exit 0
else
    echo "$0 failure: POS-scores incorrectly calculated"
    exit 1
fi
