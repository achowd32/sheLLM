import sys
import json

batch_size = int(sys.argv[1])
block_size = int(sys.argv[2])

batch_x, batch_y = [], []
i = 0
for line in sys.stdin:
    cur = [int(n) for n in line.split()]
    batch_x.append(cur[:block_size])
    batch_y.append(cur[1:block_size+1])
    i += 1
    if i % batch_size == 0:
        print(json.dumps({"batch_x": batch_x, "batch_y": batch_y}))
        batch_x, batch_y = [], []
