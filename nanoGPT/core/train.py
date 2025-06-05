import sys
import json
import torch

sys.path.append("..")
from arch import architecture

vocab_size = int(sys.argv[1])
eval_interval = int(sys.argv[2])
max_iters = int(sys.argv[3])

i = 0
for line in sys.stdin:
    batch = json.loads(line)
    xb, yb = torch.tensor(batch["batch_x"]), torch.tensor(batch["batch_y"])
    
    # load in model and optimizer
    model = architecture.GPTLanguageModel(vocab_size)
    model.load_state_dict(torch.load("../model/model.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(torch.load("../model/optimizer.pth"))
    model.train()

    # log model state
    if i % eval_interval == 0 or i == max_iters - 1:
         file_name = f"../logs/{i}.pth"
         torch.save(model.state_dict(), file_name)
         print(i, flush=True)
 
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # save model and optimizer
    torch.save(model.state_dict(), "../model/model.pth")
    torch.save(optimizer.state_dict(), "../model/optimizer.pth")
    
    #iterate
    i += 1
