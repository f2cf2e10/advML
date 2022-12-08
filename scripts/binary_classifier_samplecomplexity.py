import numpy as np
from utils.model import LogisticRegression
from adversary.solver import robust_adv_data_driven_binary_classifier
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_samples
from itertools import repeat
import multiprocessing

# Parameters
m = 1000  # each sample size
N = 1000  # number of samples
xi = 0.1  # adversarial power
r = 1  # norm_inf{x} <= 1
delta = 0.01

# Data
const = True
data = generate_synthetic_linear_model_with_uniform_distr_samples(0.0, m, N, with_const=const)
logm0 = LogisticRegression(np.random.rand(data[0][0].get('x').shape[1]), with_const=const)

r_rob = 0.5 + (0.5 - (1 - 2. ** 0.5 * xi) ** 2 / 2)
rhs_extra_term = 4. / xi * (r ** 2 / m) ** 0.5 + (np.log(np.log2(2 * r / xi)) / m) ** 0.5 + (
        np.log(2 / delta) / 2 / m) ** 0.5


def check_sample_inequality(r_hat: float, r_rob: float, rhs_extra_term: float) -> bool:
    return (r_rob - r_hat) <= rhs_extra_term


def approx_error(r_hat: float, xi: float) -> float:
    r_rob = xi / 2.0 * (2 * 2.0 ** 0.5 - xi)
    return r_hat - r_rob


def task(k, data_k, xi, r_rob, rhs_extra_term):
    print("Starting {}".format(k))
    _, r_hat = robust_adv_data_driven_binary_classifier(xi, data_k)
    # return_dict = {k: check_sample_inequality(r_hat, r_rob, rhs_extra_term)}
    # return_dict = {k: approx_error(r_hat, xi)}
    return_dict = {k: r_hat}
    print("Finished {}".format(k))
    return return_dict


multiprocessing.freeze_support()
with multiprocessing.Pool(8) as pool:
    acc = pool.starmap(task, zip(range(len(data)), data, repeat(xi), repeat(r_rob), repeat(rhs_extra_term)))

accuracy_results.describe()
accuracy_results.hist()
