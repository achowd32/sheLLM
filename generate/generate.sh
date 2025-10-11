#!/bin/bash
cd "$(dirname "$0")"

# handle arguments
model_dir="$1"
output_toks="$2"
output_prompt="$3"
num_outputs="$4"

# generate text samples
echo -e "${BLUE}Generating text samples...${RESET}"
# TODO: tokenize the prompt, feed it to generate.js, detokenize the output
# do the above as many times as necessary to generate the appropriate number of output samples,
# each of which should be stored in its own file in outputs/
