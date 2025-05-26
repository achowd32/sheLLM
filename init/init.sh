#!/bin/bash

python3 init/init_data.py $1 | while read line; do
    echo "$line" >> "tmp/train.txt"
done
python3 init_model.py