#!/bin/bash
cd "$(dirname "$0")"

# handle arguments
train_file="$1"
val_file="$2"
model_file="$3"

# number of samples to train on
num_samples=$((MAX_ITERS * BATCH_SIZE))

# initiate core pipeline
echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./tokenize.sh $num_samples "$train_file" | 
./chunk.py $BATCH_SIZE $BLOCK_SIZE |
./train.py $LOG_INTERVAL $LEARNING_RATE "$model_file" |
./log.sh "$train_file" "$val_file"
