#!/bin/bash

sed 's/\(. \)/\1\n/g' | awk '{printf("%c", $1)}' 
