import sys
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("..")
from arch import architecture

# load arguments
file_name = sys.argv[1]
eval_iters = int(sys.argv[2])

# load and initialize model
vocab_size = 128
model = architecture.GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load(file_name))

loss_sum = 0 
for line in sys.stdin: # read in one batch at a time
    # load from json and convert to tensor
    data = json.loads(line)
    X, Y = torch.tensor(data["batch_x"]), torch.tensor(data["batch_y"])

    # calculate loss and add to loss_sum
    logits, loss = model(X, Y)
    loss_sum += loss

# print average loss
print(loss_sum / eval_iters)
