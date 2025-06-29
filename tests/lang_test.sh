#!/bin/bash

cd "$(dirname "$0")"
set -o allexport
source "test_params.py"
set +o allexport

sample="This is a sentence with no errors. This es a sentence with one error."
langscore=$(python3 ../eval/lang_eval.py "$sample" 2>&1 | grep "Numerrors: " | awk '{print $2}')

if [ "$langscore" -eq 1 ]; then
    echo "$0 success: langscore correctly calculated"
    exit 0
else
    echo "$0 failure: langscore incorrectly calculated"
    exit 1
fi
