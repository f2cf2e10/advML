from functools import reduce
from typing import List, Tuple
from cvxopt import matrix, solvers
import random
import numpy as np
from adversary.attack import fast_gradient_dual_norm_method, projected_gradient_ascent
from utils.types import Model, Loss, Data, Norm


def adversarial_gd_fast_attack(loss: Loss, model0: Model, data: List[Data], tol: float, xi: float, norm: Norm,
                               eta: float = 0.05) -> Tuple[Model, List[float]]:
    training_loss = []
    delta = np.inf
    model_ = model0.copy()
    model = model0.copy()
    i = 0
    # adv_data = [fast_gradient_dual_norm_method(loss, model, norm, data_i, xi) for data_i in data]
    while delta > tol:
        adv_data = [fast_gradient_dual_norm_method(loss, model, norm, data_i, xi) for data_i in data]
        theta = model_.get_theta() - eta * loss.dtheta_batch(model, adv_data)
        model.set_theta(theta)
        delta = np.abs(loss.f_batch(model, adv_data) - loss.f_batch(model_, adv_data))
        training_loss += [loss.f_batch(model, adv_data)]
        model_ = model.copy()
        i += 1
    print(i, delta)
    return model, training_loss


def adversarial_gd_pgd_attack(loss: Loss, model0: Model, data: List[Data], tol: float, xi: float, norm: Norm,
                              eta1: float = 0.05, eta2: float = 0.05, k: int = 10) -> Tuple[Model, List[float]]:
    training_loss = []
    delta = np.inf
    model_ = model0.copy()
    model = model0.copy()
    i = 0
    # adv_data = [fast_gradient_dual_norm_method(loss, model, norm, data_i, xi) for data_i in data]
    while delta > tol:
        adv_data = [projected_gradient_ascent(loss, model, norm, data_i, xi, eta2, k) for data_i in data]
        theta = model_.get_theta() - eta1 * loss.dtheta_batch(model, adv_data)
        model.set_theta(theta)
        delta = np.abs(loss.f_batch(model, adv_data) - loss.f_batch(model_, adv_data))
        training_loss += [loss.f_batch(model, adv_data)]
        model_ = model.copy()
        i += 1
    print(i, delta)
    return model, training_loss


def adversarial_trades(loss: Loss, model0: Model, data: List[Data], proj: Norm, tol: float, xi: float,
                       lamb: float, batch_size: int, eta1: float = 0.025, eta2: float = 0.05, K: int = 20,
                       sigma: float = 0.001, max_iter: int = 2500) -> Tuple[Model, List[float]]:
    """
    TRADES algorithm from
    Theoretically Principled Trade-off between Robustness and Accuracy, El Gahoui's paper

    :param loss:
    :param model0:
    :param data:
    :param proj:
    :param xi:
    :param lamb:
    :param tol:
    :param eta1:
    :param eta2:
    :param k:
    :param batch_size:
    :param sigma:
    :return:
    """
    training_loss = []
    delta = np.inf
    model_ = model0.copy()
    model = model0.copy()
    j = 0
    while (delta > tol) & (j < max_iter):
        batch = np.random.choice(data, batch_size, replace=False)
        x_prime = []
        for i in range(batch_size):
            x_i = batch[i].get('x')
            m = len(x_i) if not model.with_const else len(x_i) - 1
            x_i_prime = x_i.copy()
            x_i_prime[0: m] = x_i_prime[0:m] + np.random.normal(0, sigma, m)
            for k in range(K):
                x_i_prime[0:m] = proj.proj(x_i_prime[0:m] +
                                           eta1 * np.sign(loss.dx(model, {'x': x_i_prime, 'y': model.value(x_i)}))[0:m],
                                           x_i[0:m], xi)
            x_prime += [x_i_prime]
        theta = model_.get_theta() - eta2 * reduce(lambda x, y: x + y, [
            loss.dtheta(model, batch[j]) +
            loss.dtheta(model, {'x': batch[j].get('x'), 'y': model.value(x_prime[j])}) / lamb
            #loss.dtheta(model, {'x': x_prime[j], 'y': model.value(batch[j].get('x'))}) / lamb
            for j in range(len(batch))]) / len(batch)
        model.set_theta(theta)
        delta = np.abs(loss.f_batch(model, data) - loss.f_batch(model_, data))
        training_loss += [loss.f_batch(model, data)]
        model_ = model.copy()
        j += 1
    print(j, delta)
    return model, training_loss


def _build_constraint_matrix(xi: float, data: List[Data], norm_bound: float = 1.0) -> matrix:
    def f(s: Data) -> matrix:
        x = s.get('x').tolist()
        y = s.get('y')
        return matrix([-(y * x_i) / (xi * norm_bound) for x_i in x])

    return matrix([[f(s)] for s in data]).T


def robust_adv_data_driven_binary_classifier(xi: float, data: List[Data], norm_bound: float = 1.0) -> np.array:
    """
    Our Algorithm, worst case

    :param xi:
    :param data:
    :return:
    """

    def _zero(m: int, d: int) -> matrix:
        return matrix(np.zeros([m, d]))

    def _one(m: int, d: int) -> matrix:
        return matrix(np.ones([m, d]))

    def _id(m: int) -> matrix:
        return matrix(np.eye(m))

    m = len(data)
    d = len(data[0].get('x'))
    c = matrix([0.0] * d + [1. / m] * m + [0.0] * d)
    b = matrix([[-2.0] * m + [0.0] * m + [0.0] * d + [0.0] * d + [norm_bound]])
    S = _build_constraint_matrix(xi, data)
    A = matrix([[S, _zero(m, d), _id(d), -_id(d), _zero(1, d)],
                [-_id(m), -_id(m), _zero(d, m), _zero(d, m), _zero(1, m)],
                [_zero(m, d), _zero(m, d), -_id(d), -_id(d), _one(1, d)]])
    sol = solvers.lp(c, A, b)
    w = sol['x'][0:d]
    return w, sol['primal objective']
