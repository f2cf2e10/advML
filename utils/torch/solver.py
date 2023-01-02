import torch
from torch import nn


def training(data, model, loss_fn, opt=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss, total_err = 0., 0.
    n = len(data.dataset)
    for X, y in data:
        X, y = X.to(device), y.to(device)
        yp = model(nn.Flatten()(X))[:, 0]
        loss = loss_fn(yp, y.float())
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += ((yp > 0) * (y == 0) + (yp <= 0) * (y == 1)).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / n, total_loss / n
