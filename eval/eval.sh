#!/bin/bash
max_tokens=500

echo -e "${BLUE}Generating text with trained model...${RESET}"
echo -ne "" | ./encode.sh | python3 generate.py $max_tokens | ./decode.sh > sample.txt
echo "Text saved to eval/sample.txt"

sample=$(cat sample.txt)
reference=$(head -c +"$max_tokens" ../data/val.txt)

echo -e "${BLUE}Performing language evaluation...${RESET}"
numerrors=$(python3 lang_eval.py "$sample")
echo "This text has ${numerrors} errors"

echo -e "${BLUE}Performing BERT evaluation...${RESET}"
python3 bert_eval.py "$sample" "$reference" 2>&1 | grep "BERTScore"

echo -e "${BLUE}Performing Part-Of-Speech evaluation...${RESET}"
python3 pos_eval.py "$sample" "$reference"
