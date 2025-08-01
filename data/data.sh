#!/bin/bash
cd "$(dirname $0)"

# download training data if it doesn't exist
echo -e "${BLUE}Initializing training data...${RESET}"
curl -o data.txt "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

#clean data (remove non-ASCII)
touch tmp.txt
LC_ALL=C tr -cd '\0-\177' < data.txt > tmp.txt
mv tmp.txt data.txt

# split into train and validation data
length=$(wc -c < data.txt | grep -oE "[0-9]+")
split=$((length * 9 / 10))
head -c "$split" data.txt > train.txt
tail -c +"$((split + 1))" data.txt > val.txt
