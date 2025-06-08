#!/bin/bash

echo -e "${BLUE}Generating text with trained model...${RESET}"
echo -ne "" | ./encode.sh | python3 generate.py | ./decode.sh > sample.txt
echo "Text saved to eval/sample.txt"

echo -e "${BLUE}Performing language evaluation...${RESET}"
numerrors=$(python3 lang_eval.py "$(cat sample.txt)")
echo "This text has ${numerrors} errors"
