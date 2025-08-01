#!/bin/bash
num_samples=$((EVAL_ITERS * BATCH_SIZE))
val_data="../data/val.txt"

while read log; do
    # calculate train loss
    tloss=$(./tokenize.sh $num_samples |
    ./chunk.js $BATCH_SIZE $BLOCK_SIZE | 
    ./log.js "../logs/${log}" 2>/dev/null)

    # calculate validation loss
    vloss=$(./tokenize.sh $num_samples $val_data |
    ./chunk.js $BATCH_SIZE $BLOCK_SIZE |
    ./log.js "../logs/${log}" 2>/dev/null)

    # print to stdout
    echo "At training interval ${log} — train loss: ${tloss}, validation loss: ${vloss}"
done
