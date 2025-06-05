import sys
import json
import torch

sys.path.append("..")
from arch import architecture

encodings = json.loads(sys.argv[1])
vocab_size = int(sys.argv[2])
prompt = sys.argv[3]

stoi = encodings["stoi"]
itos = encodings["itos"]
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[str(i)] for i in l]) 

model = architecture.GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load("../model/model.pth"))
model.eval()

context = torch.tensor([encode(prompt)], dtype=torch.long) 
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
