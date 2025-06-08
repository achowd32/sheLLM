import sys
import json
import torch

sys.path.append("..")
from arch import architecture

prompt = sys.argv[1]
vocab_size = 128

model = architecture.GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load("../model/model.pth"))
model.eval()

#context = torch.tensor([encode(prompt)], dtype=torch.long) 
context = torch.zeros((1, 1), dtype=torch.long)
tokens = model.generate(context, max_new_tokens=500)[0].tolist()
print(" ".join(str(t) for t in tokens))
