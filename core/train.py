import sys
import json
import torch
import os
import time

sys.path.append("..")
from arch import architecture

vocab_size = 128
eval_interval = int(sys.argv[1])
max_iters = int(sys.argv[2])

i = 0
for line in sys.stdin:
    batch = json.loads(line)
    xb, yb = torch.tensor(batch["batch_x"]), torch.tensor(batch["batch_y"])
    
    # load in model and optimizer
    while os.path.getsize("../model.pth") == 0:
         time.sleep(0.25)
    checkpoint = torch.load("../model.pth")
    open("../model.pth", "w").close()

    model = architecture.GPTLanguageModel(vocab_size)
    model.load_state_dict(checkpoint['model_sd'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['opt_sd'])
    model.train()

    # log model state
    if i % eval_interval == 0 or i == max_iters - 1:
         file_name = f"../logs/{i}.pth"
         torch.save(model.state_dict(), file_name)
         print(i, file = sys.stderr, flush=True)
 
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # save model and optimizer
    save_dict = {'model_sd': model.state_dict(), 'opt_sd': optimizer.state_dict()}
    torch.save(save_dict, sys.stdout.buffer)
    sys.stdout.buffer.flush()

    #iterate
    i += 1
