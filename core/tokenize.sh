#!/bin/bash

#set variables
file_name=${2:-"../data/train.txt"}
char_count=$(cat $file_name | wc -c | tr -d ' ')
upper_offset=$((char_count - BLOCK_SIZE))
sample_size=$((BLOCK_SIZE + 1))
num_samples=$1

#loop over number of samples
i=0
offset=1
while [ $i -lt $num_samples ]; do # CHECK LOGIC
    # in case we need to loop around
    if [ $offset -gt $upper_offset ]; then
        end_sample=$((char_count - offset + 1))
        start_sample=$((sample_size - end_sample))
        tail -c +"$offset" "$file_name" | head -c +"$end_sample" | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' ' 
        offset=1
        tail -c +"$offset" "$file_name" | head -c +"$start_sample" | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' ' 
        echo ""
        ((i+=1))
        ((offset+=start_sample))
        continue
    fi

    tail -c +"$offset" "$file_name" | head -c +"$sample_size" | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' ' #sample and tokenize
    echo "" #add newline

    ((i+=1)) #iterate
    ((offset+=sample_size)) #increase offset
done
