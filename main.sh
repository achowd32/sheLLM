#!/bin/bash

# may pivot to docker later
source ../shvenv/bin/activate

mkdir checkpoints

numfiles=${1:-5}
python3 init_data.py $numfiles | while read line; do
    echo "$line" >> "train.txt"
done

python3 init_model.py

cat train.txt | python3 tkn.py | python3 chunk.py | python3 train.py

python3 interpret.py "Who are you" >> interpret.txt
python3 interpret.py "Tell me a story" >> interpret.txt
python3 interpret.py "Generate five interesting sentences" >> interpret.txt

deactivate