#!/bin/bash

#handle flag to switch to validation data
while getopts "v" flag; do
 case $flag in
   v) # Handle the -v flag (use val.txt)
    file_name="../data/val.txt"
    ;;
 esac
done

#set variables
file_name=${file_name:-"../data/train.txt"}
char_count=$(cat $file_name | wc -c | tr -d ' ')
upper_rand=$((char_count - BLOCK_SIZE - 1)) #confirm dimensions later
data_string=$(cat $file_name)
num_samples=$1

#loop over number of samples
i=0
while [ $i -lt $num_samples ]; do
    rand=$(jot -r 1 0 $upper_rand) #get a random starting position
    echo -n "${data_string:${rand}:$((BLOCK_SIZE+1))}" | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' ' #sample and tokenize
    echo "" #add newline
    i=$((i+1)) #iterate
done
