#!/bin/bash
cd "$(dirname "$0")"

# handle arguments
model_dir="$1"
output_toks="$2"
output_prompt="$3"
num_outputs="$4"

# generate text samples
echo -e "${BLUE}Generating text samples...${RESET}"
i=0
while [ $i -lt $num_outputs ]; do
    echo "$output_prompt" | od -An -t u1 -v | xargs |
    ./generate.js "../$model_dir" $output_toks 2>/dev/null |
    sed 's/\(. \)/\1\n/g' | awk '{printf("%c", $1)}' > "../outputs/output_${i}.txt"

    ((i++))
done
