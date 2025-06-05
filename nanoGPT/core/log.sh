#!/bin/bash

while read log; do
    tloss=$(./sample.sh $((EVAL_ITERS * BATCH_SIZE)) |
    python3 tknize.py "$(cat ../model/encoding.json)" "$BLOCK_SIZE" |
    python3 chunk.py $BATCH_SIZE |
    python3 log.py "../logs/${log}.pth" $VOCAB_SIZE $EVAL_ITERS |
    grep -oE "[0-9]+\.[0-9]+")
    echo "At training interval ${log} — train loss: ${tloss}"
done
