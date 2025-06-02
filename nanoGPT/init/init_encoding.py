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
output = {"stoi": stoi, "itos": itos}
print(json.dumps(output), flush=True)