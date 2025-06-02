#!/bin/bash
BLUE='\033[1;34m'
RESET='\033[0m'

batch_size=12
block_size=64
max_iters=500
vocab_size=$(cat ../data/train.txt | sed 's/\(.\)/\1\n/g' | sort -u | wc -l | tr -d ' ')

echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./sample.sh $batch_size $block_size $max_iters |
python3 tknize.py "$(cat ../model/encoding.json)" "$block_size" |
python3 chunk.py $batch_size |
python3 train.py $vocab_size

echo -e "${BLUE}Generating text with trained model...${RESET}"
python3 generate.py "$(cat ../model/encoding.json)" "$vocab_size"