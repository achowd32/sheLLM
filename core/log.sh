#!/bin/bash

# initialize arguments
num_samples=$((LOG_ITERS * BATCH_SIZE))
train_file="$1"
val_file="$2"

# get chunked tokens
ttoks="$(./tokenize.sh $num_samples "$train_file" | ./chunk.js $BATCH_SIZE $BLOCK_SIZE)"
vtoks="$(./tokenize.sh $num_samples "$val_file" | ./chunk.js $BATCH_SIZE $BLOCK_SIZE)"

while read log; do
    # TODO: calculate train loss
    tloss=

    # TODO: calculate validation loss
    vloss=

    # print to stdout to see progress
    echo "At training interval ${log} — train loss: ${tloss}, validation loss: ${vloss}"
done
