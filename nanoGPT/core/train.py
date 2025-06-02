import sys
import json
import torch
import architecture as arch

vocab_size = int(sys.argv[1])

for line in sys.stdin:
    batch = json.loads(line)
    xb, yb = torch.tensor(batch["batch_x"]), torch.tensor(batch["batch_y"])
    
    # load in model and optimizer
    model = arch.GPTLanguageModel(vocab_size)
    model.load_state_dict(torch.load("../model/model.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(torch.load("../model/optimizer.pth"))
    model.train()

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # save model and optimizer
    torch.save(model.state_dict(), "../model/model.pth")
    torch.save(optimizer.state_dict(), "../model/optimizer.pth")
