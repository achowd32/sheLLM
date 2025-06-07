import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("..")
from arch import architecture

learning_rate = 3e-4

# create a PyTorch model
vocab_size = 128
model = architecture.GPTLanguageModel(vocab_size)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# save model and optimizer
torch.save(model.state_dict(), "../model/model.pth")
torch.save(optimizer.state_dict(), "../model/optimizer.pth")
