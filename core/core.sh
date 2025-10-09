#!/bin/bash
cd "$(dirname "$0")"

# handle arguments
train_file="$1"
val_file="$2"
model_dir="$3"
max_tokens="$4"
num_outputs="$5"

# number of samples to train on
num_samples=$((MAX_ITERS * BATCH_SIZE))

# initiate core pipeline
echo -e "${BLUE}Initiating core tokenization and training loop...${RESET}"
./tokenize.sh $num_samples "$train_file" |
./chunk.js $BATCH_SIZE $BLOCK_SIZE |
./train.js $LOG_INTERVAL $LEARNING_RATE "$model_dir" 2>/dev/null |
./log.sh  "$train_file" "$val_file"

# generate text samples
echo -e "${BLUE}Generating text samples...${RESET}"
i=0
while [ $i -lt $num_outputs ]; do
    echo "" | 
    ./generate.js "../$model_dir" $max_tokens 2>/dev/null |
    sed 's/\(. \)/\1\n/g' | awk '{printf("%c", $1)}' > "../outputs/output_${i}.txt"

    ((i++))
done
