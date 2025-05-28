#!/bin/bash
BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Running perplexity evaluation...${RESET}"
python3 perp_eval.py "$(cat prompts.txt)"

echo -e "${BLUE}Running language evaluation...${RESET}"
while IFS= read -r line || [ -n "$line" ]; do
    output=$(python3 interpret.py "$line" | tee -a ../outputs/language.txt)
    numerrors=$(python3 lang_eval.py "$output")
    echo "${line}: ${numerrors}"
done < prompts.txt

echo -e "${BLUE}Running summary evaluation...${RESET}"
while IFS= read -r line || [ -n "$line" ]; do
    prompt=$(echo -e "Summarize this:\n${line}\nSummary:")
    python3 interpret.py "$prompt" | tr '\n' ' ' >> ../outputs/summary.txt
    echo '\n' >> ../outputs/summary.txt
done < ../tmp/summary_article.txt
rouge -f ../outputs/summary.txt ../tmp/summary_highlights.txt