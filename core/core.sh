#!/bin/bash
BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}Running core tokenization and training pipeline...${RESET}"
cat ../tmp/train.txt | python3 tkn.py | python3 chunk.py | python3 train.py