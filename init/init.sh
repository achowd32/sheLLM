#!/bin/bash

BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Initializing training data...${RESET}"
python3 init_train.py $1 >> "../tmp/train.txt"

echo -e "${BLUE}Initializing summary evaluation data...${RESET}"
python3 init_summary.py "article" >> "../tmp/summary_article.txt"
python3 init_summary.py "highlights" >> "../tmp/summary_highlights.txt"

echo -e "${BLUE}Initializing model...${RESET}"
python3 init_model.py $2