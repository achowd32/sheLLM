import sys
import json
import torch

sys.path.append("..")
from arch import architecture

vocab_size = 128
filename = sys.argv[1]
max_tok = int(sys.argv[2])

model = architecture.GPTLanguageModel(vocab_size)
checkpoint = torch.load(f"../{filename}")

model.load_state_dict(checkpoint['model_sd'])
model.eval()

prompt = sys.stdin.read()
context = torch.tensor([[int(n) for n in prompt.split()]], dtype=torch.long)
if len(context[0]) == 0:
     context = torch.zeros((1, 1), dtype=torch.long)

tokens = model.generate(context, max_new_tokens=max_tok)[0].tolist()
print(" ".join(str(t) for t in tokens))
