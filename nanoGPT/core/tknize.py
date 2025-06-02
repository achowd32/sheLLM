import sys
import json

encodings = json.loads(sys.argv[1])
stoi = encodings["stoi"]
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
i = 0
while True:
    block_size = int(sys.argv[2])
    block = sys.stdin.buffer.read(block_size + 2)
    if not block:
        break
    tokens = encode(block.decode('utf-8'))
    x = tokens[:block_size]
    y = tokens[1:block_size+1]
    print(json.dumps({"x": x, "y": y}))