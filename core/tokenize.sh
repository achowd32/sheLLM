#!/bin/bash

#set variables
file_name=${2:-"../data/train.txt"} # name of training data file, defaults to data/train.txt
char_count=$(wc -c < "$file_name" | tr -d ' ') # number of characters in training data file
upper_offset=$((char_count - BLOCK_SIZE)) # offset beyond which tokenizer must wrap around text
sample_size=$((BLOCK_SIZE + 1)) # number of tokens per sample
num_samples=$1 # number of samples

#tokenization function
tokenize(){
    local head_offset=$1
    tail -c +"$offset" "$file_name" | head -c "$head_offset" | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' ' 
}

#loop over number of samples
i=0
offset=1
while [ $i -lt $num_samples ]; do 
    if [ $offset -gt $upper_offset ]; then # if the next intake requires looping around the text

        # if no text is remaining, reset to the beginning (avoids head -c 0)
        [ $offset -eq $((char_count+1)) ] && offset=1 && continue
        
        # handle offset logic
        end_sample=$((char_count - offset + 1))
        start_sample=$((sample_size - end_sample))

        # tokenize ending portion, reset, and tokenize starting portion
        tokenize $end_sample
        offset=1
        tokenize $start_sample

    else

        # simply tokenize the required number of tokens per sample
        tokenize $sample_size 

    fi

    echo "" #add newline
    ((i+=1)) #iterate
    ((offset+=sample_size)) #increase offset
done
