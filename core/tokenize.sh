#!/bin/bash

# set variables
num_samples=$1 # number of samples
sample_size=$((BLOCK_SIZE + 1)) # number of tokens per sample
file_name=${2:-"../data/train.txt"} # training data filename, defaults to data/train.txt

# get number of iterations (to avoid partial output at the end)
char_count=$(wc -c < "$file_name" | tr -d ' ') # training data file character count
poss_iters=$((char_count / BLOCK_SIZE)) # number of possible iterations in training data
[[ $num_samples -gt $poss_iters ]] && num_samples=$poss_iters

# create file descriptor
exec 3< "$file_name"

# commence reading and tokenizing loop
for i in $(seq $num_samples); do
    dd bs=$sample_size count=1 2>/dev/null <&3 | od -An -t u1 -v | xargs 2>/dev/null
done
