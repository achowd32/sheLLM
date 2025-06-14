import sys
import json
import torch

sys.path.append("..")
from arch import architecture

vocab_size = 128

# initialize arguments
eval_interval = int(sys.argv[1])
max_iters = int(sys.argv[2])
filename = sys.argv[3]

i = 0

# one line is one batch in json format; keep reading while data is streaming in
for line in sys.stdin:
    # read from json and convert to tensor
    batch = json.loads(line)
    xb, yb = torch.tensor(batch["batch_x"]), torch.tensor(batch["batch_y"])
    
    # load in model and optimizer
    checkpoint = torch.load(f"../{filename}")

    model = architecture.GPTLanguageModel(vocab_size)
    model.load_state_dict(checkpoint['model_sd'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['opt_sd'])
    model.train()

    # print iteration to logging pipeline 
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
    save_dict = {'model_sd': model.state_dict(), 'opt_sd': optimizer.state_dict()}
    torch.save(save_dict, f"../{filename}")

    #iterate
    i += 1
