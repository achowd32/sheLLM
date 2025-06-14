import sys
import json

# initialize arguments
batch_size = int(sys.argv[1])
block_size = int(sys.argv[2])

# initialize loop variables
batch_x, batch_y = [], []
i = 0

# each line is one sample, we keep reading while data is streaming in
for line in sys.stdin:
    # convert to integer
    cur = [int(n) for n in line.split()]

    # slice into correct blocks
    batch_x.append(cur[:block_size])
    batch_y.append(cur[1:block_size+1])

    # iterate
    i += 1
    
    # if we have collected 'batch_size' many samples, we print to stdout
    if i % batch_size == 0:
        print(json.dumps({"batch_x": batch_x, "batch_y": batch_y}), flush = True)
        batch_x, batch_y = [], []
