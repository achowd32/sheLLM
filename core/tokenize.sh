#!/bin/bash

# set variables
num_samples=$1
train_file="../data/$2"
sample_size=$((BLOCK_SIZE + 1))

# get number of iterations (to avoid partial output at the end)
char_count=$(wc -c < "$train_file" | tr -d ' ') # training data file character count
poss_iters=$((char_count / sample_size)) # number of possible iterations in training data
[[ $num_samples -gt $poss_iters ]] && num_samples=$poss_iters

# TODO: take the right number of samples and tokenize each one
# HINT: look into the dd command and the od command
# this file should output a fixed number of space-separated tokens on each line
