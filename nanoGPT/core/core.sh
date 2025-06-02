#!/bin/bash
BLUE='\033[1;34m'
RESET='\033[0m'

batch_size=12
block_size=64
max_iters=3

echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./sample.sh $batch_size $block_size $max_iters |
python3 tokenize.py "$(cat ../model/encoding.json)" $block_size |
python3 chunk.py $batch_size |
python3 train.py