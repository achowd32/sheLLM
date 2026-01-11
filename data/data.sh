#!/bin/bash
cd "$(dirname "$0")"

# get filenames
data_file="$1"
train_file="$2"
val_file="$3"

# download training data
echo -e "${BLUE}Initializing training data...${RESET}"
if [ -f "$data_file" ]; then
    echo "${data_file} already initialized -- skipping download"
else
    curl -o "$data_file" "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
fi

# TODO: clean data (remove non-ASCII characters) and insert into a temporary txt file
# __start_solution__
tmp_file="/tmp/tmp.txt"
LC_ALL=C tr -cd '\0-\177' < "$data_file" > "$tmp_file"
# __end_solution__

# TODO: split into training and validation data
# recall that the first 90% (rounded down) of all data should go into the training data file,
# and all other data should go into the validation data file
# __start_solution__
length=$(wc -c < "$tmp_file" | grep -oE "[0-9]+")
split=$((length * 9 / 10))
head -c "$split" "$tmp_file" > "$train_file"
tail -c +"$((split + 1))" "$tmp_file" > "$val_file"
rm "$tmp_file"
# __end_solution__
