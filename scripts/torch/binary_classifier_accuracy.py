import multiprocessing
import time
from itertools import repeat
from collections import ChainMap

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pylab as plt

from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_sample, \
    generate_synthetic_linear_model_with_cap_normal_distr_sample, \
    generate_synthetic_linear_model_with_uniform_distr_samples
from utils.torch.solver import training


def scatter_plot(model, title, data_x, data_y):
    plt.figure()
    plt.scatter(data_x[:, 0], data_x[:, 1], marker="o", c=data_y, s=35)
    plt.scatter(data_x[:, 0], data_x[:, 1], marker="+",
                c=[+1.0 if model(torch.Tensor(x_i)) > 0 else -1.0 for x_i in data_x], s=35)
    plt.title(title)
    plt.show()


# Parameters
# model
sigma = 0.1
tol = 1E-5
n_train = 1000
n_test = 500
n_paths = 1000
n = n_train + n_test
norm_bound = 1.0
# adversarial power
xis = [0.1, 0.2, 0.3]
np.random.seed(1771)
torch.manual_seed(7777)
accuracy = []

data = generate_synthetic_linear_model_with_uniform_distr_samples(sigma, n, n_paths, 0.75, 0.1)


def task(k, data_k, n_paths, n_train, n_test, xi, norm_bound):
    x = np.array([d.get('x') for d in data_k])
    y = np.array([d.get('y') for d in data_k])
    d = len(x[0])
    t0 = time.time()
    accuracy_i = []

    train_x = torch.Tensor(x[0: n_train])
    train_y = (torch.Tensor(y[0: n_train]) + 1) / 2
    train_set = TensorDataset(train_x, train_y)
    train_data = DataLoader(train_set, batch_size=100, shuffle=True)

    test_x = torch.Tensor(x[n_train: (n_train + n_test)])
    test_y = (torch.Tensor(y[n_train: (n_train + n_test)]) + 1) / 2
    test_set = TensorDataset(test_x, test_y)
    test_data = DataLoader(test_set, batch_size=100, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()

    model = nn.Linear(d, 1)
    delta = np.Inf
    previous_train_loss = np.Inf
    while delta > tol:
        train_err, train_loss = training(train_data, model, loss_fn, True)
        test_err, test_loss = training(test_data, model, loss_fn)
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
    accuracy_i += [1 - test_err]

    model_robust_fgsm = nn.Linear(d, 1)
    delta = np.Inf
    previous_train_loss = np.Inf
    while delta > tol:
        train_err, train_loss = adversarial_training_fast_gradient_sign_method(
            train_data, model_robust_fgsm, loss_fn, adv_loss_fn, True, xi=xi, norm_bound=norm_bound)
        test_err, test_loss = training(test_data, model_robust_fgsm, loss_fn)
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
    accuracy_i += [1 - test_err]

    model_robust_pgd = nn.Linear(d, 1)
    delta = np.Inf
    previous_train_loss = np.Inf
    while delta > tol:
        train_err, train_loss = adversarial_training_projected_gradient_descent(
            train_data, model_robust_pgd, loss_fn, adv_loss_fn, True, xi=xi, norm_bound=norm_bound)
        test_err, test_loss = training(test_data, model_robust_pgd, loss_fn)
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
    accuracy_i += [1 - test_err]

    model_robust_trades = nn.Linear(d, 1)
    delta = np.Inf
    previous_train_loss = np.Inf
    while delta > tol:
        train_err, train_loss = adversarial_training_trades(
            train_data, model_robust_trades, loss_fn, adv_loss_fn, True, xi=xi)
        test_err, test_loss = training(test_data, model_robust_trades, loss_fn)
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
    accuracy_i += [1 - test_err]

    our_model, adv_our_loss, adv_our_err = robust_adv_data_driven_binary_classifier(train_data, xi=xi, norm_bound=norm_bound)
    test_err, test_loss = training(test_data, our_model, loss_fn)
    accuracy_i += [1 - test_err]
    print("{}/{} - {}".format(k, n_paths, time.time() - t0))
    return accuracy_i

multiprocessing.freeze_support()
for xi in xis:
    with multiprocessing.Pool(8) as pool:
        acc = pool.starmap(task, zip(range(len(data)), data, repeat(n_paths), repeat(n_train), repeat(n_test), repeat(xi), repeat(norm_bound)))

    columns = ["Training", "Adv training - FGSM", "Adv training - PGD", "TRADES", "Our model"]
    accuracy_results = pd.DataFrame(acc, columns=columns)
    accuracy_results.to_csv('./accuracy_torch_1000_paths_xi_' + str(xi) + '.csv')