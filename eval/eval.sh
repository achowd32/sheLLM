#!/bin/bash
BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Running perplexity evaluation...${RESET}"
python3 perp_eval.py "$(cat prompts.txt)"

echo -e "${BLUE}Running language evaluation...${RESET}"
while IFS= read -r line || [ -n "$line" ]; do
    output=$(python3 interpret.py "$line" | tee -a interpret.txt)
    numerrors=$(python3 lang_eval.py "$output")
    echo "${line}: ${numerrors}"
done < prompts.txt
