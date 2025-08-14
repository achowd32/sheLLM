#!/bin/bash
cd "$(dirname "$0")"

# initialize arguments
val_file="$1"
model_file="$2"
sample_file="$3"
which_eval="$4"

# initialize variables
num_evals=1
prompt=""
max_tokens=500

echo -e "${BLUE}Performing evaluations...${RESET}"

iters=0
eval_sum=0
while [ $iters -lt $num_evals ]; do
    # generate sample
    ./generate.sh "../${model_file}.pth" "$prompt" "$max_tokens" > "$sample_file"
   
    # load the sample text and reference text into variables
    sample=$(cat "$sample_file")
    reference=$(head -c "$max_tokens" "../data/$val_file")

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
    ((iters++))
done 

eval_avg=$(echo "scale=5; $eval_sum / $num_evals" | bc)
echo "Score: ${eval_avg}"
