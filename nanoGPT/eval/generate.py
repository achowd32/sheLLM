import sys
import json
import torch

sys.path.append("..")
from arch import architecture

vocab_size = int(sys.argv[2])

encodings = json.loads(sys.argv[1])
itos = encodings["itos"]
decode = lambda l: ''.join([itos[str(i)] for i in l])

model = architecture.GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load("../model/model.pth"))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))