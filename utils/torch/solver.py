import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader


def training(data: DataLoader, model: Module, loss_fn: Module, opt: bool = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss, total_err = 0., 0.
    n = len(data.dataset)
    for x_i, y_i in data:
        x, y = x_i.to(device), y_i.to(device)
        yp = model(nn.Flatten()(x))[:, 0]
        loss = loss_fn(yp, y.float())
        if opt:
            optimizer = optim.SGD(model.parameters(), lr=1e-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_err += ((yp > 0) * (y == 0) + (yp <= 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x_i.shape[0]
    return total_err / n, total_loss / n
