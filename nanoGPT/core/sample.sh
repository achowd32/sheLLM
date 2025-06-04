#!/bin/bash
char_count=$(cat ../data/train.txt | wc -c | tr -d ' ')
upper_rand=$((char_count - BLOCK_SIZE - 1)) #confirm dimensions later
data_string=$(cat ../data/train.txt)
num_samples=$1

i=0
while [ $i -lt $num_samples ]; do
    rand=$(jot -r 1 0 $upper_rand)
    echo "${data_string:${rand}:$((BLOCK_SIZE+1))}"
    i=$((i+1))
done
