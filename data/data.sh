#!/bin/bash
cd "$(dirname $0)"

# get filenames
data_file="data.txt"
train_file="train.txt"
val_file="val.txt"

# download training data if it doesn't exist
echo -e "${BLUE}Initializing training data...${RESET}"
if [ -f "$data_file" ]; then
    echo "${data_file} already initialized -- skipping"
else
    curl -o "$data_file" "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
fi

#clean data (remove non-ASCII)
touch tmp.txt
LC_ALL=C tr -cd '\0-\177' < "$data_file" > tmp.txt
mv tmp.txt "$data_file"

# split into train and validation data
length=$(wc -c < "$data_file" | grep -oE "[0-9]+")
split=$((length * 9 / 10))
head -c "$split" "$data_file" > "$train_file"
tail -c +"$((split + 1))" "$data_file" > "$val_file"
