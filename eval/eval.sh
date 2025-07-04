#!/bin/bash
model_file="../model.keras"
prompt=""
max_tokens=500
num_evals=1

lang_sum=0
prec_sum=0
rec_sum=0
f1_sum=0
pos_sum=0

echo -e "${BLUE}Performing evaluations...${RESET}"

iters=0
while [ $iters -lt $num_evals ]; do
    # generate text sample and save it; will be used for evaluations
    echo -ne "$prompt" | ./encode.sh | python3 generate.py $model_file $max_tokens | ./decode.sh > sample.txt
    
    # load the sample text and reference text into variables
    sample=$(cat sample.txt)
    reference=$(head -c "$max_tokens" ../data/val.txt)

    # perform a simple language evaluation: number of language errors in sample
    lang_score=$(python3 lang_eval.py "$sample" 2>&1 | grep "Numerrors: " | awk '{print $2}')
    ((lang_sum += lang_score))

    # perform an evaluation of the semantic similarity between sample and validation data
    bertscores=$(python3 bert_eval.py "$sample" "$reference" 2>&1 | grep "BERTScores:" | awk '{print $2, $3, $4}')
    read precision recall f1 <<< "$bertscores"
    prec_sum=$(echo "$prec_sum + $precision" | bc -l)
    rec_sum=$(echo "$rec_sum + $recall" | bc -l)
    f1_sum=$(echo "$f1_sum + $f1" | bc -l)

    # perform an evaluation of the syntactic similarity between sample and validation data
    pos_score=$(python3 pos_eval.py "$sample" "$reference" 2>&1 | grep "Syntactic similarity" | awk '{print $3}')
    pos_sum=$(echo "$pos_sum + $pos_score" | bc -l)

    # iterate
    ((iters+=1))
done 

lang_avg=$(echo "scale=5; $lang_sum / $num_evals" | bc)
prec_avg=$(echo "scale=5; $prec_sum / $num_evals" | bc)
rec_avg=$(echo "scale=5; $rec_sum / $num_evals" | bc)
f1_avg=$(echo "scale=5; $f1_sum / $num_evals" | bc)
pos_avg=$(echo "scale=5; $pos_sum / $num_evals" | bc)

echo -e "${BLUE}Printing evaluation results, averaged over ${num_evals} iterations...${RESET}"
echo "Language evaluation: ${lang_avg} language errors detected on average" | tee results.txt
echo -e "Semantic similarity evaluation (BERT):\nPrecision—${prec_avg}\nRecall Score—${rec_avg}\nF1 Score—${f1_avg}" | tee -a results.txt
echo "Syntactic similarity evaluation (Part-Of-Speech): ${pos_avg}" | tee -a results.txt
