#!/bin/bash
cd "$(dirname "$0")"

# get filenames
data_file="$1"
train_file="$2"
val_file="$3"

# download training data if it doesn't exist
curl -o "$data_file" "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# TODO: clean data (remove non-ASCII characters)

# TODO: split into training and validation data
# recall that the first 90% (rounded down) of all data should go into the training data file,
# and all other data should go into the validation data file
