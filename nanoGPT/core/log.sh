#!/bin/bash

while read log; do
    echo $log | grep -oE "\d+"
    ./sample.sh $((EVAL_ITERS * BATCH_SIZE)) |
    python3 tknize.py "$(cat ../model/encoding.json)" "$BLOCK_SIZE" |
    python3 chunk.py $BATCH_SIZE |
    python3 log.py $log $VOCAB_SIZE $EVAL_ITERS
done
