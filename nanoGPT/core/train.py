import sys
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

model.load_state_dict(torch.load("model.pth"))
optimizer.load_state_dict(torch.load("optimizer.pth"))

for line in sys.stdin:
    batch = json.loads(line)
    xb, yb = batch["batch_x"], batch["batch_y"]

# evaluate the loss
logits, loss = model(xb, yb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()