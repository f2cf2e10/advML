import numpy as np
import torch
from cvxopt import matrix, solvers
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from adversary.torch.attack import fast_gradient_sign_method, projected_gradient_descent_method
from utils.torch.norm import Linf
from utils.torch.types import Norm


def adversarial_training_fast_gradient_sign_method(data: DataLoader, model: Module, loss_fn: Module,
                                                   adv_loss_fn: Module, opt: bool = False, xi: float = 0.2, 
                                                   norm_bound: float = np.Inf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss, total_err = 0., 0.
    n = len(data.dataset)
    for x_i, y_i in data:
        x, y = x_i.to(device), y_i.to(device)
        x_adv = torch.clamp(fast_gradient_sign_method(model, adv_loss_fn, nn.Flatten()(x), y, xi), 
                            min=-norm_bound, max=norm_bound)
        y_hat = model(x_adv)[:, 0]
        loss = loss_fn(y_hat, y.float())
        if opt:
            optimizer = optim.SGD(model.parameters(), lr=1e-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_err += ((y_hat > 0) * (y == 0) + (y_hat < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / n, total_loss / n


def adversarial_training_projected_gradient_descent(data: DataLoader, model: Module, loss_fn: Module,
                                                    adv_loss_fn: Module, opt: bool = False, proj: Norm = Linf,
                                                    xi: float = 0.2, norm_bound: float = np.Inf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss, total_err = 0., 0.
    n = len(data.dataset)
    for x_i, y_i in data:
        x, y = x_i.to(device), y_i.to(device)
        x_adv = torch.clamp(projected_gradient_descent_method(model, adv_loss_fn, proj, nn.Flatten()(x), y, xi), 
                            min=-norm_bound, max=norm_bound)
        y_hat = model(x_adv)[:, 0]
        loss = loss_fn(y_hat, y.float())
        if opt:
            optimizer = optim.SGD(model.parameters(), lr=1e-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_err += ((y_hat > 0) * (y == 0) + (y_hat < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / n, total_loss / n


def adversarial_training_trades(data: DataLoader, model: Module, loss_fn: Module, adv_loss_fn: Module,
                                opt: bool = False, proj: Norm = Linf, xi: float = 0.2, lamb: float = 1.0,
                                k: int = 20, eta1: float = 0.025, sigma: float = 0.001, norm_bound:float = np.Inf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss, total_err = 0., 0.
    n = len(data.dataset)
    for x_i, y_i in data:
        x, y = x_i.to(device), y_i.to(device)
        x_0 = nn.Flatten()(x) if x.ndim > 1 else x
        y_x_0 = Variable(((model(x_0)[:, 0]).sign() + 1) / 2)
        x_adv = Variable(torch.clamp(x_0 + sigma * torch.randn_like(x_0), 
                                     min=-norm_bound, max=norm_bound), requires_grad=True)
        adv_optimizer = optim.SGD([x_adv], lr=1e-1)
        for k_i in range(k):
            adv_optimizer.zero_grad()
            loss = adv_loss_fn(model(x_adv)[:, 0], y_x_0)
            loss.backward()
            x_adv = Variable(proj.proj(x_adv + eta1 * torch.sign(x_adv.grad.detach()), x_0, xi), requires_grad=True)
        y_hat = model(x_0)[:, 0]
        y_hat_0_1 = (y_hat.sign() + 1) / 2
        y_hat_adv = Variable(model(x_adv)[:, 0])
        loss = loss_fn(y_hat, y.float()) + loss_fn(y_hat_adv, y_hat_0_1) / lamb
        if opt:
            optimizer = optim.SGD(model.parameters(), lr=1e-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_err += ((y_hat > 0) * (y == 0) + (y_hat < 0) * (y == 1)).sum().item()
        total_loss += loss.item() * x.shape[0]
    return total_err / n, total_loss / n


def robust_adv_data_driven_binary_classifier(data: DataLoader, xi: float = 0.2, norm_bound: float = 1.0) -> np.array:
    def _build_constraint_matrix(xi: float, data: Dataset, norm_bound: float = 1.0) -> matrix:
        vals = []
        for x_i, y_i in data:
            x = (nn.Flatten()(x_i) if x_i.ndim > 1 else x_i).numpy()
            y = y_i.numpy() if isinstance(y_i, torch.Tensor) else y_i
            for i in range(len(y)):
                vals += [-(2 * y[i] - 1) * np.hstack([x[i, :], 1.0]) / (xi * norm_bound)]
            # if model.const:
            #    vals += [-(2 * y - 1) * np.hstack([x, 1.0]) / (xi * norm_bound)]
            # else:
            #    vals += [-(2 * y - 1) * x / (xi * norm_bound)]
        vals = np.array(vals).astype(np.double)
        return matrix(vals)

    def _zero(m: int, d: int) -> matrix:
        return matrix(np.zeros([m, d]))

    def _one(m: int, d: int) -> matrix:
        return matrix(np.ones([m, d]))

    def _id(m: int) -> matrix:
        return matrix(np.eye(m))

    S = _build_constraint_matrix(xi, data)
    m, d = np.shape(S)
    c = matrix([0.0] * d + [1. / m] * m + [0.0] * d)
    b = matrix([[-2.0] * m + [0.0] * m + [0.0] * d + [0.0] * d + [norm_bound]])
    A = matrix([[S, _zero(m, d), _id(d), -_id(d), _zero(1, d)],
                [-_id(m), -_id(m), _zero(d, m), _zero(d, m), _zero(1, m)],
                [_zero(m, d), _zero(m, d), -_id(d), -_id(d), _one(1, d)]])
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, A, b)
    w = sol['x'][0:d]
    model = nn.Linear(d, 1)
    model.weight = nn.Parameter(torch.Tensor(np.array(w[0:(d - 1)]).T))
    model.bias = nn.Parameter(torch.Tensor(np.array([w[d - 1]])))
    loss = sol['primal objective']

    total_err = 0
    for x_i, y_i in data:
        x = nn.Flatten()(x_i) if x_i.ndim > 1 else x_i
        y = y_i
        y_hat = model(x).T[0]
        total_err += (((y_hat > 0) * (y == 0) + (y_hat < 0) * (y == 1)) * 1.0).sum().item()
    return model, total_err / m, loss
