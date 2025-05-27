#!/bin/bash

echo "--------------------------------"
echo "Running perplexity evaluation..."
python3 perp_eval.py

echo "--------------------------------"
echo "Running language evaluation..."
while IFS= read -r line || [ -n "$line" ]; do
    output=$(python3 interpret.py "$line")
    numerrors=$(python3 lang_eval.py "$output")
    echo "${line}: ${numerrors}"
done < prompts.txt
