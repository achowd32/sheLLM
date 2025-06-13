#!/bin/bash
filename="model.pth"

echo -e "${BLUE}Initializing training data...${RESET}"
if [ -f "train.txt" ]; then
    echo "train.txt already initialized -- skipping"
else
    curl -o ../data/data.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    
    #split into train and validation data
    string=$(cat ../data/data.txt)
    length=${#string}
    split=$(( $length * 9 / 10 ))
    echo "${string:0:${split}}" > ../data/train.txt
    echo "${string:${split}:${length}}" > ../data/val.txt
fi

echo -e "${BLUE}Initializing model...${RESET}"
python3 init_model.py $LEARNING_RATE $filename 
