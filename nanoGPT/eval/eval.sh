#!/bin/bash

echo -e "${BLUE}Generating text with trained model...${RESET}"
python3 generate.py "$(cat ../model/encoding.json)" "$VOCAB_SIZE"