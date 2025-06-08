#!/bin/bash
num_samples=$((EVAL_ITERS * BATCH_SIZE))
while read log; do
    tloss=$(./sample.sh $num_samples | 
    python3 chunk.py $BATCH_SIZE $BLOCK_SIZE |
    python3 log.py "../logs/${log}.pth" $EVAL_ITERS |
    grep -oE "[0-9]+\.[0-9]+")

    vloss=$(./sample.sh $num_samples -v |
    python3 chunk.py $BATCH_SIZE $BLOCK_SIZE |
    python3 log.py "../logs/${log}.pth" $EVAL_ITERS |
    grep -oE "[0-9]+\.[0-9]+")

    echo "At training interval ${log} — train loss: ${tloss}, validation loss: ${vloss}"
done
