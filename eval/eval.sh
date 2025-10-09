#!/bin/bash
cd "$(dirname "$0")"

# initialize arguments
val_file="$1"
which_eval="$2"
output_len="$3"

echo -e "${BLUE}Performing evaluations...${RESET}"

eval_sum=0
reference=$(head -c "$output_len" "../data/$val_file")

for sample_file in ../outputs/*; do
    # load the sample text and reference text into variables
    sample=$(cat "$sample_file")
    
    # run evaluation based on which_eval argument
    if [[ "$which_eval" == "lang" ]]; then
        eval_score=$(./lang_eval.py "$sample")
    elif [[ "$which_eval" == "bert" ]]; then
        eval_score=$(./bert_eval.py "$sample" "$reference")
    elif [[ "$which_eval" == "posp" ]]; then
        eval_score=$(./pos_eval.py "$sample" "$reference")
    else
        continue
    fi

    eval_sum=$(echo "$eval_sum + $eval_score" | bc -l)
done 

total_files=$(ls -1 ../outputs/ | wc -l)
eval_avg=$(echo "scale=5; $eval_sum / $total_files" | bc)
echo "Evaluation score: ${eval_avg}"
