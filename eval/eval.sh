#!/bin/bash

prompts=(cat prompts.txt)
echo "$prompts"

echo "Running perplexity evaluation..."
python3 perp_eval.py

echo "Running language evaluation..."
