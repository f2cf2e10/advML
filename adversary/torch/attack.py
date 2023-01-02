import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import Dataset

from utils.torch.norm import Linf
from utils.torch.types import Norm


def fast_gradient_sign_method(model: Module, loss_fn: Module, x: Dataset, y: Dataset, xi: float):
    return fast_gradient_dual_norm_method(model, loss_fn, Linf, x, y, xi)


def fast_gradient_dual_norm_method(model: Module, loss_fn: Module, norm: Norm, x: Dataset, y: Dataset, xi: float):
    delta = torch.zeros_like(x, requires_grad=True)
    y_hat = model(x + delta)
    if y_hat.shape != y.shape:
        y_hat = y_hat[:, 0]
    loss = loss_fn(y_hat, y.float())
    grad = torch.autograd.grad(loss, [delta])[0]
    return x + norm.dual_dx(grad) * xi


def projected_gradient_descent_method(model: Module, loss_fn: Module, proj: Norm, x: Dataset, y: Dataset, xi: float,
                                      alpha: float = 0.025, k: int = 20):
    return projected_gradient_descent_norm_method(model, loss_fn, Linf, proj, x, y, xi, alpha, k)


def projected_gradient_descent_norm_method(model: Module, loss_fn: Module, norm: Norm, proj: Norm, x: Dataset,
                                           y: Dataset, xi: float, alpha: float = 0.025, k: int = 100):
    x_0 = torch.clone(x)
    x_i = torch.clone(x)
    x_i.requires_grad = True
    for k_i in range(k):
        y_hat = model(x_i.float())
        if y_hat.shape != y.shape:
            y_hat = y_hat[:, 0]
        loss = loss_fn(y_hat, y.float())
        grad = torch.autograd.grad(loss, [x_i])
        x_i = Variable(proj.proj(x_i + alpha * norm.dual_dx(grad[0]), x_0, torch.Tensor([xi])), requires_grad=True)
    return x_i
