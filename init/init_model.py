import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("..")
from arch import architecture

# initialize arguments
learning_rate = float(sys.argv[1])
filename = sys.argv[2]

# create a PyTorch model
vocab_size = 128
model = architecture.GPTLanguageModel(vocab_size)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# save model and optimizer
save_dict = {'model_sd': model.state_dict(), 'opt_sd': optimizer.state_dict()}
torch.save(save_dict, f"../{filename}")
