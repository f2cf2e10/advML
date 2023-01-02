import numpy as np
import torch
from torch import nn

from adversary.torch.attack import fast_gradient_sign_method, fast_gradient_dual_norm_method, \
    projected_gradient_descent_method
from utils.torch.norm import Linf, L2, L1


def test_fast_gradient_sign_method():
    d = np.random.randint(2, 100)
    loss = nn.BCEWithLogitsLoss()
    model = nn.Linear(d, 1)
    x = torch.Tensor(np.random.randn(d))
    y = torch.Tensor([np.random.randint(0, 2)])
    xi = np.random.random(1)
    x_adv = fast_gradient_sign_method(model, loss, x, y, xi)
    norm = Linf()
    assert (torch.abs(norm.norm(x - x_adv) - xi) <= 1E-7)


def test_fast_gradient_dual_norm_method_L2():
    d = np.random.randint(2, 100)
    loss = nn.BCEWithLogitsLoss()
    model = nn.Linear(d, 1)
    x = torch.Tensor(np.random.randn(d))
    y = torch.Tensor([np.random.randint(0, 2)])
    xi = np.random.random(1)
    norm = L2()
    x_adv = fast_gradient_dual_norm_method(model, loss, norm, x, y, xi)
    assert (torch.abs(norm.norm(x - x_adv) - xi) <= 1E-7)


def test_projected_gradient_descent_method():
    d = np.random.randint(2, 100)
    loss = nn.BCEWithLogitsLoss()
    model = nn.Linear(d, 1)
    x = torch.Tensor(np.random.randn(d))
    y = torch.Tensor([np.random.randint(0, 2)])
    xi = 0.5 * np.random.random(1)
    proj = Linf()
    norm = Linf()
    x_adv = projected_gradient_descent_method(model, loss, proj, x, y, xi)
    assert norm.norm(x - x_adv) <= torch.Tensor(xi) + 1E-7


def test_projected_gradient_descent_norm_L2_method():
    d = np.random.randint(2, 100)
    loss = nn.BCEWithLogitsLoss()
    model = nn.Linear(d, 1)
    x = torch.Tensor(np.random.randn(d))
    y = torch.Tensor([np.random.randint(0, 2)])
    xi = 0.5 * np.random.random(1)
    proj = L2()
    norm = Linf()
    x_adv = projected_gradient_descent_method(model, loss, proj, x, y, xi)
    assert norm.norm(x - x_adv) <= torch.Tensor(xi) + 1E-7


def test_projected_gradient_descent_norm_L1_method():
    d = np.random.randint(2, 100)
    loss = nn.BCEWithLogitsLoss()
    model = nn.Linear(d, 1)
    x = torch.Tensor(np.random.randn(d))
    y = torch.Tensor([np.random.randint(0, 2)])
    xi = 0.5 * np.random.random(1)
    proj = L1()
    norm = Linf()
    x_adv = projected_gradient_descent_method(model, loss, proj, x, y, xi)
    assert norm.norm(x - x_adv) <= torch.Tensor(xi) + 1E-7
