import time
import multiprocessing
from collections import ChainMap
from itertools import repeat, cycle
from typing import Callable
import numpy as np
import pandas as pd
from utils.loss import CrossEntropy
from utils.model import LogisticRegression
from utils.norm import Linf
from adversary.solver import adversarial_gd_fast_attack, robust_adv_data_driven_binary_classifier, \
    adversarial_gd_pgd_attack, adversarial_trades
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_samples, \
    generate_synthetic_linear_model_with_cap_normal_distr_samples
from utils.types import Data

# Parameters
sigma = 0.1  # noise
m = 1000  # each sample size
N = 600  # number of samples
xi = [0.001, 0.01, 0.1]  # adversarial power

# Data
const = True
data = generate_synthetic_linear_model_with_uniform_distr_samples(sigma, m, N, with_const=const)
logm0 = LogisticRegression(np.random.rand(data[0][0].get('x').shape[0]), with_const=const)


def calculate_accuracy(data: Data, model: Callable[[np.ndarray], float]):
    x = [d.get('x') for d in data]
    y = [d.get('y') for d in data]
    correct_classification = 0
    for i in range(len(y)):
        y_model = model(x[i])
        if y_model == y[i]:
            correct_classification += 1
    return correct_classification / m


def task(k, data_k, logm0, xi, m):
    print("Starting {}".format(k))
    ret = []
    logm_adv_fgsm, _ = adversarial_gd_fast_attack(CrossEntropy(), logm0, data_k, 1E-5, xi, Linf)
    logm_adv_pgd, _ = adversarial_gd_pgd_attack(CrossEntropy(), logm0, data_k, 1E-5, xi, Linf)
    lamb = 0.1
    logm_adv_trades, _ = adversarial_trades(CrossEntropy(), logm0, data_k, Linf, 1E-5, xi, lamb, int(m / 10))
    w, _ = robust_adv_data_driven_binary_classifier(xi, data_k)
    ret = [calculate_accuracy(data_k, lambda x: 1.0 if logm_adv_fgsm.value(x) >= 0.5 else 0.0),
           calculate_accuracy(data_k, lambda x: 1.0 if logm_adv_pgd.value(x) >= 0.5 else 0.0),
           calculate_accuracy(data_k, lambda x: 1.0 if logm_adv_trades.value(x) >= 0.5 else 0.0),
           calculate_accuracy(data_k, lambda x: 1.0 if x.dot(w) >= 0.0 else 0.0)]
    return_dict = {k: ret, 'xi': xi}
    print("Finished {}".format(k))
    return return_dict


t0 = time.time()
multiprocessing.freeze_support()
with multiprocessing.Pool(8) as pool:
    acc = pool.starmap(task, zip(range(len(data)), data, repeat(logm0), cycle(xi), repeat(m)))

time.time() - t0
columns = ["Adv training - FGSM", "Adv training - PGD", "TRADES", "Our model"]
robustness_results = {xi_i : pd.DataFrame([list(a.values())[0] for a in acc if a.get('xi')==xi_i], 
columns=columns) for xi_i in xi}

robustness_results.describe()
robustness_results.hist()
