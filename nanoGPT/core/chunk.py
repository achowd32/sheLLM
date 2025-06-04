import sys
import json

batch_size = int(sys.argv[1])
batch_x, batch_y = [], []
i = 0
for line in sys.stdin:
    cur = json.loads(line)
    batch_x.append(cur["x"])
    batch_y.append(cur["y"])
    i += 1
    if i % batch_size == 0:
        print(json.dumps({"batch_x": batch_x, "batch_y": batch_y}))
        batch_x, batch_y = [], []
