#!/bin/bash
filename='model'
num_samples=$((MAX_ITERS * BATCH_SIZE))

# initiate core pipeline
echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./tokenize.sh $num_samples | 
python3 chunk.py $BATCH_SIZE $BLOCK_SIZE |
python3 train.py $EVAL_INTERVAL $MAX_ITERS $filename #|
#./log.sh 
