#!/bin/bash

# initialize arguments
num_samples=$((LOG_ITERS * BATCH_SIZE))
train_file="$1"
val_file="$2"

# get chunked tokens
ttoks="$(./tokenize.sh $num_samples "$train_file" | ./chunk.js $BATCH_SIZE $BLOCK_SIZE)"
vtoks="$(./tokenize.sh $num_samples "$val_file" | ./chunk.js $BATCH_SIZE $BLOCK_SIZE)"

while read log; do
    # calculate train loss
    tloss=$(echo "$ttoks" | ./log.js "../logs/${log}" 2>/dev/null)

    # calculate validation loss
    vloss=$(echo "$vtoks" | ./log.js "../logs/${log}" 2>/dev/null)

    # print to stdout
    echo "At training interval ${log} — train loss: ${tloss}, validation loss: ${vloss}"
done
