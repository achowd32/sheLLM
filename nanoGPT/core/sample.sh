#!/bin/bash
batch_size="$1"
block_size="$2"
max_iters="$3"
char_count=$(cat ../data/train.txt | wc -c | tr -d ' ')
upper_rand=$((char_count - block_size - 1)) #confirm dimensions later
data_string=$(cat ../data/train.txt)

i=0
while [ $i -lt $((max_iters * batch_size)) ]; do
    rand=$(jot -r 1 0 $upper_rand)
    echo "${data_string:${rand}:$((block_size+1))}"
    i=$((i+1))
done