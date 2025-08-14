#!/bin/bash
model_file="$1"
prompt="$2"
max_tokens="$3"

# tokenization encode and decoding functions
encode() {
    echo -ne "$1" | od -An -t u1 -v | xargs
}

decode() {
    sed 's/\(. \)/\1\n/g' | awk '{printf("%c", $1)}' 
}

# generate text sample and save it; will be used for evaluations
encode "$prompt" | ./generate.py "$model_file" $max_tokens | decode
