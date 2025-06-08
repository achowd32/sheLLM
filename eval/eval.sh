#!/bin/bash
echo -e "${BLUE}Generating text with trained model...${RESET}"
python3 generate.py "Who are you?" | ./decode.sh
echo ""
