#!/bin/bash
filename="model.keras"

# download training data if it doesn't exist
echo -e "${BLUE}Initializing training data...${RESET}"
if [ -f "../data/data.txt" ]; then
    echo "data.txt already initialized -- skipping"
else
    curl -o ../data/data.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

#clean data (remove non-ASCII)
touch ../data/tmp.txt
LC_ALL=C tr -cd '\0-\177' < ../data/data.txt > ../data/tmp.txt
mv ../data/tmp.txt ../data/data.txt

# split into train and validation data
length=$(wc -c < ../data/data.txt | grep -oE "[0-9]+")
split=$((length * 9 / 10))
head -c "$split" ../data/data.txt > ../data/train.txt
tail -c +"$((split + 1))" ../data/data.txt > ../data/val.txt

# initialize model
echo -e "${BLUE}Initializing model...${RESET}"
python3 init_model.py $LEARNING_RATE $filename 
