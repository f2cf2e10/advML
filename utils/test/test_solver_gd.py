import numpy as np

from utils.loss import L2, CrossEntropy
from utils.model import LinearRegression, LogisticRegression
from utils.solver import gd
from utils import norm as norm


def test_gd_lm_l2_loss():
    n = 100
    a = np.random.rand()
    b = np.random.rand()
    e = np.random.randn(n)
    x = np.vstack([np.arange(n) / 100., np.ones(n)]).T
    y = np.array([a, b]).dot(x.T) + e
    data = [{'x': x[i, :], 'y': y[i]} for i in range(n)]
    tol = 1E-8
    model0 = LinearRegression(np.array([1.0, 0.0]))
    model_opt = LinearRegression(np.array([a, b]))
    loss = L2()
    lm, delta = gd(loss, model0, data, tol)
    ols = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    lm.get_theta() - ols
    assert delta[-2] - delta[-1] <= tol


def test_gd_lm_l2_loss_linf_const():
    n = 100
    a = np.random.rand()
    b = np.random.rand()
    e = np.random.randn(n)
    x = np.vstack([np.arange(n) / 100., np.ones(n)]).T
    y = np.array([a, b]).dot(x.T) + e
    data = [{'x': x[i, :], 'y': y[i]} for i in range(n)]
    tol = 1E-8
    model0 = LinearRegression(np.array([1.0, 0.0]))
    model_opt = LinearRegression(np.array([a, b]))
    loss = L2()
    proj = norm.Linf
    const = 1.0
    lm, delta = gd(loss, model0, data, tol, proj=proj, constraint=const)
    assert delta[-2] - delta[-1] <= tol
    assert proj.norm(lm.get_theta()) <= const


def test_gd_lm_l2_loss_l2_const():
    n = 100
    a = np.random.rand()
    b = np.random.rand()
    e = np.random.randn(n)
    x = np.vstack([np.arange(n) / 100., np.ones(n)]).T
    y = np.array([a, b]).dot(x.T) + e
    data = [{'x': x[i, :], 'y': y[i]} for i in range(n)]
    tol = 1E-8
    model0 = LinearRegression(np.array([1.0, 0.0]))
    model_opt = LinearRegression(np.array([a, b]))
    loss = L2()
    proj = norm.L2
    const = 1.0
    lm, delta = gd(loss, model0, data, tol, proj=proj, constraint=const)
    assert delta[-2] - delta[-1] <= tol
    assert proj.norm(lm.get_theta()) <= const


def test_gd_lm_l2_loss_l1_const():
    n = 100
    a = np.random.rand()
    b = np.random.rand()
    e = np.random.randn(n)
    x = np.vstack([np.arange(n) / 100., np.ones(n)]).T
    y = np.array([a, b]).dot(x.T) + e
    data = [{'x': x[i, :], 'y': y[i]} for i in range(n)]
    tol = 1E-8
    model0 = LinearRegression(np.array([0.5, 0.5]))
    model_opt = LinearRegression(np.array([a, b]))
    loss = L2()
    proj = norm.L1
    const = 1.0
    lm, delta = gd(loss, model0, data, tol, proj=proj, constraint=const)
    assert delta[-2] - delta[-1] <= tol
    assert proj.norm(lm.get_theta()) <= const

def test_gd_log_ce_loss():
    n = 100
    a = np.random.rand()
    b = np.random.rand() - 0.5
    e = np.random.randn(n)
    x = np.vstack([np.random.rand(n), np.ones(n)]).T
    y = np.random.rand(n)
    z = [1 if np.array([a, b]).dot(x[i, :]) > y[i] else 0 for i in range(n)]
    data = [{'x': x[i, :], 'y': z[i]} for i in range(n)]
    tol = 1E-6
    model0 = LogisticRegression(np.array([1.0, 0.0]))
    model_opt = LogisticRegression(np.array([a, b]))
    loss = CrossEntropy()
    logit, delta = gd(loss, model0, data, tol)
    assert delta[-2] - delta[-1] <= tol
