#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

sample="This is a sample sentence, which is identical to the other."
reference="This is a sample sentence, which is identical to the other."

bertscore=$(python3 ../eval/bert_eval.py "$sample" "$reference")

if [[ "$bertscore" == "1.0000" ]]; then
    echo "$0 success: BERTScores correctly calculated"
    exit 0
else
    echo "$0 failure: BERTScores incorrectly calculated"
    exit 1
fi
