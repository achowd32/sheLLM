#!/bin/bash

# initialize arguments
train_file="$1"
val_file="$2"
num_samples=$((LOG_ITERS * BATCH_SIZE))

# get chunked tokens
ttoks="$(./tokenize.sh $num_samples "$train_file" | ./chunk.py $BATCH_SIZE $BLOCK_SIZE)"
vtoks="$(./tokenize.sh $num_samples "$val_file" | ./chunk.py $BATCH_SIZE $BLOCK_SIZE)"

while read log; do
    # calculate train loss
    tloss=$(echo "$ttoks" |
    ./log.py "../logs/${log}.pth" $LOG_ITERS |
    grep -oE "[0-9]+\.[0-9]+")

    # calculate validation loss
    vloss=$(echo "$vtoks" |
    ./log.py "../logs/${log}.pth" $LOG_ITERS |
    grep -oE "[0-9]+\.[0-9]+")

    # print to stdout
    echo "At training interval ${log} — train loss: ${tloss}, validation loss: ${vloss}"
done
