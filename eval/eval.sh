#!/bin/bash
prompt=""
model_file="model.pth"
max_tokens=500

# generate text sample and save it; will be used for evaluations
echo -e "${BLUE}Generating text with trained model...${RESET}"
echo -ne "$prompt" | ./encode.sh | python3 generate.py $model_file $max_tokens | ./decode.sh > sample.txt
echo "Text saved to eval/sample.txt"

# load the sample text and reference text into variables
sample=$(cat sample.txt)
reference=$(head -c +"$max_tokens" ../data/val.txt)

# perform a simple language evaluation: number of language errors in sample
echo -e "${BLUE}Performing language evaluation...${RESET}"
python3 lang_eval.py "$sample" 2>&1 | grep "This text has"

# perform an evaluation of the semantic similarity between sample and validation data
echo -e "${BLUE}Performing BERT evaluation...${RESET}"
python3 bert_eval.py "$sample" "$reference" 2>&1 | grep "BERTScore"

# perform an evaluation of the syntactic similarity between sample and validation data
echo -e "${BLUE}Performing Part-Of-Speech evaluation...${RESET}"
python3 pos_eval.py "$sample" "$reference" 2>&1 | grep "Syntactic similarity" 
