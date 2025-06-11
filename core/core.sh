#!/bin/bash
filename='model.pth'

echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./sample.sh $((MAX_ITERS * BATCH_SIZE)) | 
python3 chunk.py $BATCH_SIZE $BLOCK_SIZE |
python3 train.py $EVAL_INTERVAL $MAX_ITERS $filename |
./log.sh 
