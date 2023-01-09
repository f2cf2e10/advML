import multiprocessing
import time
from itertools import repeat

import numpy as np
import torch
from scipy import special
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pylab as plt

from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_sample, \
    generate_synthetic_linear_model_with_cap_normal_distr_sample, \
    generate_synthetic_linear_model_with_uniform_distr_samples, \
    generate_synthetic_linear_model_with_cap_normal_distr_samples
from utils.torch.solver import training


def scatter_plot(model, title, data_x, data_y):
    plt.figure()
    plt.scatter(data_x[:, 0], data_x[:, 1], marker="o", c=data_y, s=35)
    plt.scatter(data_x[:, 0], data_x[:, 1], marker="+",
                c=[+1.0 if model(torch.Tensor(x_i)) > 0 else -1.0 for x_i in data_x], s=35)
    plt.title(title)
    plt.show()


# Parameters
n_train = 1000  # each sample size
n_test = 500
n_paths = 1000  # number of samples
xi = 0.1  # adversarial power
r = 1  # norm_inf{x} <= 1
delta = 0.1
distr = 'normal'

rhs_extra_term = 4. / xi * (r ** 2 / n_train) ** 0.5 + (np.log(np.log2(2 * r / xi)) / n_train) ** 0.5 + (
        np.log(2 / delta) / 2 / n_train) ** 0.5

# Data
const = True
if distr == 'uniform':
    data = generate_synthetic_linear_model_with_uniform_distr_samples(0.0, n_train, n_paths, a=0., b=0.)
    r_rob = xi
else:
    std = 0.1
    data = generate_synthetic_linear_model_with_cap_normal_distr_samples(0.0, n_train, n_paths, a=0., b=0., std=std)
    r_rob = 2 * special.erf(xi / std)


def task(k, data_k, n_paths, n_train, n_test, xi):
    x = np.array([d.get('x') for d in data_k])
    y = np.array([d.get('y') for d in data_k])
    t0 = time.time()

    train_x = torch.Tensor(x[0: n_train])
    train_y = (torch.Tensor(y[0: n_train]) + 1) / 2
    train_set = TensorDataset(train_x, train_y)
    train_data = DataLoader(train_set, batch_size=100, shuffle=True)

    test_x = torch.Tensor(x[n_train: (n_train + n_test)])
    test_y = (torch.Tensor(y[n_train: (n_train + n_test)]) + 1) / 2
    test_set = TensorDataset(test_x, test_y)
    test_data = DataLoader(test_set, batch_size=100, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()

    our_model, adv_our_err, adv_our_loss = robust_adv_data_driven_binary_classifier(train_data, xi=xi)
    print("{}/{} - {}".format(k, n_paths, time.time() - t0))
    return adv_our_loss

acc = []
for k in range(n_paths):
    acc += [task(k, data[k], n_paths, n_train, n_test, xi)]

multiprocessing.freeze_support()
with multiprocessing.Pool(6) as pool:
    acc = pool.starmap(task, zip(range(len(data)), data, repeat(n_paths), repeat(n_train), repeat(n_test), repeat(xi)))
