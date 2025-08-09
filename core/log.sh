#!/bin/bash
num_samples=$((EVAL_ITERS * BATCH_SIZE))
train_file="$1"
val_file="$2"

while read log; do
    # calculate train loss
    tloss=$(./tokenize.sh $num_samples "$train_file" |
    ./chunk.js $BATCH_SIZE $BLOCK_SIZE | 
    ./log.js "../logs/${log}" 2>/dev/null)

    # calculate validation loss
    vloss=$(./tokenize.sh $num_samples "$val_file" |
    ./chunk.js $BATCH_SIZE $BLOCK_SIZE |
    ./log.js "../logs/${log}" 2>/dev/null)

    # print to stdout
    echo "At training interval ${log} — train loss: ${tloss}, validation loss: ${vloss}"
done
