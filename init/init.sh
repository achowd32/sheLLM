#!/bin/bash

BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Initializing training data...${RESET}"
python3 init_data.py $1 | while read line; do
    echo "$line" >> "../tmp/train.txt"
done

echo -e "${BLUE}Initializing model...${RESET}"
python3 init_model.py