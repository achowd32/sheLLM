#!/bin/bash
cd "$(dirname "$0")"

# handle arguments
train_file="$1"
val_file="$2"
model_dir="$3"

# number of samples to train on
num_samples=$((MAX_ITERS * BATCH_SIZE))

# initiate core pipeline
echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./tokenize.sh $num_samples "$train_file" | 
./chunk.js $BATCH_SIZE $BLOCK_SIZE |
./train.js $EVAL_INTERVAL $LEARNING_RATE "$model_dir" 2>/dev/null |
./log.sh  "$train_file" "$val_file"
