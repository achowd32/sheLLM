#!/bin/bash
BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./sample.sh $((MAX_ITERS * BATCH_SIZE)) |
python3 tknize.py "$(cat ../model/encoding.json)" "$BLOCK_SIZE" |
python3 chunk.py $BATCH_SIZE |
python3 train.py $VOCAB_SIZE $EVAL_INTERVAL |
./log.sh
