import numpy as np

from adversary.attack import fast_gradient_sign_method, fast_gradient_dual_norm_method, projected_gradient_ascent
from utils.loss import L2 as L2Loss
from utils.model import LinearRegression
from utils.norm import Linf, L2, L1


def test_fast_gradient_sign_method():
    loss = L2Loss()
    lm = LinearRegression(np.random.randn(2))
    data = {'x': np.random.randn(2), 'y': 1.0}
    xi = 0.1
    norm = Linf()
    data_prime = fast_gradient_sign_method(loss, lm, data, xi)
    assert (np.abs(norm.norm(data_prime.get('x') - data.get('x')) - xi) <= 1E-7)


def test_fast_gradient_dual_norm_method_L2():
    loss = L2Loss()
    lm = LinearRegression(np.random.randn(2))
    data = {'x': np.random.randn(2), 'y': 1.0}
    xi = 0.1
    norm = L2()
    data_prime = fast_gradient_dual_norm_method(loss, lm, norm, data, xi)
    assert (np.abs(norm.norm(data_prime.get('x') - data.get('x')) - xi) <= 1E-7)


def test_projected_gradient_ascent_L2():
    loss = L2Loss()
    lm = LinearRegression(np.random.randn(2))
    data = {'x': np.random.randn(2), 'y': 1.0}
    xi = 0.1
    proj = L2()
    data_prime = projected_gradient_ascent(loss, lm, proj, data, xi, 0.01, 100)
    assert (np.abs(np.abs(proj.norm(data_prime.get('x') - data.get('x'))) - xi) <= 1E-7)


def test_projected_gradient_ascent_Linf():
    loss = L2Loss()
    lm = LinearRegression(np.random.randn(2))
    data = {'x': np.random.randn(2), 'y': 1.0}
    xi = 0.1
    proj = Linf()
    data_prime = projected_gradient_ascent(loss, lm, proj, data, xi, 0.01, 100)
    assert (np.abs(np.abs(proj.norm(data_prime.get('x') - data.get('x'))) - xi) <= 1E-7)


def test_projected_gradient_ascent_L1():
    loss = L2Loss()
    lm = LinearRegression(np.random.randn(2))
    data = {'x': np.random.randn(2), 'y': 1.0}
    xi = 0.1
    proj = L1()
    data_prime = projected_gradient_ascent(loss, lm, proj, data, xi, 0.01, 100)
    assert (np.abs(np.abs(proj.norm(data_prime.get('x') - data.get('x'))) - xi) <= 1E-7)
