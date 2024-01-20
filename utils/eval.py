from torch import nn
import torch
import numpy as np
import scipy as sp


def RoMa(delta, xi, x_y, n, model):
    x = x_y[0]
    y = 2 * x_y[1] - 1
    x_0 = nn.Flatten(0, x.dim() - 1)(x)
    hic = np.zeros(n)
    for i in range(n):
        x_i = torch.clamp(x_0 + torch.FloatTensor(x_0.shape).uniform_(-xi, xi), -1, 1)
        y_i = model(x_i).detach().numpy()[0]
        prob = 1. / (1 + np.exp(-np.abs(y_i)))
        hic[i] = 1 - prob if np.sign(y) == np.sign(y_i) else prob
    box_cox = sp.stats.boxcox([h for h in hic if h > 0.0])
    mu = np.mean(box_cox[0])
    sigma = np.std(box_cox[0])
    delta_box_cox = sp.stats.boxcox(delta, box_cox[1])
    return sp.stats.norm.cdf((delta_box_cox - mu) / sigma)
