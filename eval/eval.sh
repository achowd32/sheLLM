#!/bin/bash
cd "$(dirname "$0")"

# initialize arguments
val_file="$1"
output_len="$2"

echo -e "${BLUE}Performing evaluations...${RESET}"

eval_sum=0
reference=$(head -c "$output_len" "../data/$val_file")

for sample_file in ../outputs/*; do
    # load the sample text and reference text into variables
    sample=$(cat "$sample_file")
    eval_score=$(./bert_eval.py "$sample" "$reference")
    eval_sum=$(echo "$eval_sum + $eval_score" | bc -l)
done 

total_files=$(ls -1 ../outputs/ | wc -l)
eval_avg=$(echo "scale=5; $eval_sum / $total_files" | bc)
echo "Evaluation score: ${eval_avg}"
