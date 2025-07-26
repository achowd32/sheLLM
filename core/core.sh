#!/bin/bash
filename='model'
num_samples=$((MAX_ITERS * BATCH_SIZE))

# initiate core pipeline
echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./tokenize.sh $num_samples | 
./chunk.js $BATCH_SIZE $BLOCK_SIZE |
./train.js $EVAL_INTERVAL $MAX_ITERS $LEARNING_RATE $filename |
./log.sh 
