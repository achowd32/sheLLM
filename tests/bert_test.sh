#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

sample="This is a sample sentence, which is identical to the other."
reference="This is a sample sentence, which is identical to the other."

bertscores=$(python3 ../eval/bert_eval.py "$sample" "$reference" 2>&1 | grep "BERTScores:" | awk '{print $2, $3, $4}')
read precision recall f1 <<< "$bertscores"

if [[ "$precision" == "1.0000" && "$recall" == "1.0000" && "$f1" == "1.0000" ]]; then
    echo "$0 success: BERTScores correctly calculated"
    exit 0
else
    echo "$0 failure: BERTScores incorrectly calculated"
    exit 1
fi
