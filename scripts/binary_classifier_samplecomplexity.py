from collections import ChainMap

import numpy as np
import pandas as pd
from scipy import special

from utils.model import LogisticRegression
from adversary.solver import robust_adv_data_driven_binary_classifier
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_samples, \
    generate_synthetic_linear_model_with_cap_normal_distr_samples
from itertools import repeat
import multiprocessing

# Parameters
m = 1000  # each sample size
N = 1000  # number of samples
xi = 0.1  # adversarial power
r = 1  # norm_inf{x} <= 1
delta = 0.1
distr = 'normal'

rhs_extra_term = 4. / xi * (r ** 2 / m) ** 0.5 + (np.log(np.log2(2 * r / xi)) / m) ** 0.5 + (
        np.log(2 / delta) / 2 / m) ** 0.5

# Data
const = True
if distr == 'uniform':
    data = generate_synthetic_linear_model_with_uniform_distr_samples(0.0, m, N, with_const=const, a=0., b=0.)
    r_rob = xi
else:
    std = 0.1
    data = generate_synthetic_linear_model_with_cap_normal_distr_samples(0.0, m, N, with_const=const, a=0., b=0.,
                                                                         std=std)
    r_rob = 2 * special.erf(xi / std)


def task(k, data_k, xi):
    print("Starting {}".format(k))
    _, r_hat = robust_adv_data_driven_binary_classifier(xi, data_k)
    return_dict = {k: r_hat}
    print("Finished {}".format(k))
    return return_dict


multiprocessing.freeze_support()
with multiprocessing.Pool(8) as pool:
    acc = pool.starmap(task, zip(range(len(data)), data, repeat(xi)))

columns = ["R_hat"]
sample_complexity = pd.DataFrame.from_dict(dict(ChainMap(*acc)), orient='index', columns=columns)

sample_complexity.describe()
sample_complexity.hist()
