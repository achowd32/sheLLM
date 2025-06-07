#!/bin/bash
while getopts "v" flag; do
 case $flag in
   v) # Handle the -v flag (use validation)
    file_name="../data/val.txt"
    ;;
 esac
done
file_name=${file_name:-"../data/train.txt"}
char_count=$(cat $file_name | wc -c | tr -d ' ')
upper_rand=$((char_count - BLOCK_SIZE - 1)) #confirm dimensions later
data_string=$(cat $file_name)
num_samples=$1

i=0
while [ $i -lt $num_samples ]; do
    rand=$(jot -r 1 0 $upper_rand)
    echo -n "${data_string:${rand}:$((BLOCK_SIZE+1))}" | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' '
    echo ""
    i=$((i+1))
done
