import sys
import json
import torch

sys.path.append("..")
from arch import architecture

# initialize arguments
eval_interval = int(sys.argv[1])
learning_rate = float(sys.argv[2])
vocab_size = 128

# initialize model and optimizer
model = architecture.GPTLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model.train()

i = 0
# one line is one batch in json format; keep reading while data is streaming in
for line in sys.stdin:
    # read from json and convert to tensor
    batch = json.loads(line)
    xb, yb = torch.tensor(batch["batch_x"]), torch.tensor(batch["batch_y"])

    # print iteration to logging pipeline 
    if i % eval_interval == 0:
         file_name = f"../logs/{i}.pth"
         torch.save(model.state_dict(), file_name)
         print(i, flush=True)
 
    # evaluate the loss and train
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    #iterate
    i += 1

# save model
torch.save(model.state_dict(), f"../logs/{i}.pth")
print(i, flush=True)
