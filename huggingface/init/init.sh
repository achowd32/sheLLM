#!/bin/bash

BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Initializing training data...${RESET}"
if [ -f "../data/train.txt" ]; then
    echo "Training data exists -- skipping"
else
    python3 init_train.py $1 >> "../data/train.txt"
fi

scripts=(init_lang_perp.py init_lang_perp.py init_summary.py init_summary.py)
args=(instruction output text summary)
files=(language_eval.txt perplexity_eval.txt summary_article.txt summary_highlights.txt)

echo -e "${BLUE}Initializing evaluation data...${RESET}"
for ((i=0; i < ${#scripts[@]}; i++)); do
    if [ -f "../data/${files[i]}" ]; then
        echo "${files[i]} data exists -- skipping"
    else
        echo "Initializing ${files[i]}"
        python3 "${scripts[i]}" "${args[i]}" >> "../data/${files[i]}"
    fi
done

echo -e "${BLUE}Initializing model...${RESET}"
python3 init_model.py $2