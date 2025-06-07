#!/bin/bash

fold -w 65 | od -An -t u1 -v | grep -oE "[0-9]+" | tr '\n' ' '
