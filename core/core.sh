#!/bin/bash

cat tmp/train.txt | python3 tkn.py | python3 chunk.py | python3 train.py