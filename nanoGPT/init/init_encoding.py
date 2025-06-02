import sys
import json
# here are all the unique characters that occur in this text
stoi = {}
itos = {}
iter = 0
for line in sys.stdin:
    char = line[0]
    stoi[char] = iter
    itos[iter] = char
    iter += 1
print(json.dumps(stoi), flush=True)
print(json.dumps(itos), flush=True)
#encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
#decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string