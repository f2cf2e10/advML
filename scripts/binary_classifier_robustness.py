import time
import multiprocessing
from collections import ChainMap
from itertools import repeat
from typing import Callable
import numpy as np
import pandas as pd
from utils.loss import CrossEntropy
from utils.model import LogisticRegression
from utils.norm import Linf
from adversary.solver import adversarial_gd_fast_attack, robust_adv_data_driven_binary_classifier, \
    adversarial_gd_pgd_attack, adversarial_trades
from data.linear_data_generator import generate_synthetic_linear_model_samples
from utils.types import Data

# Parameters
sigma = 0.1  # noise
m = 1000  # each sample size
N = 1  # number of samples
xi = [0.005, 0.05, 0.5]  # adversarial power


def calculate_accuracy(data: Data, model: Callable[[np.ndarray], float]):
    x = [d.get('x') for d in data]
    y = [d.get('y') for d in data]
    correct_classification = 0
    for i in range(len(y)):
        y_model = model(x[i])
        if y_model == y[i]:
            correct_classification += 1
    return correct_classification / m


data = generate_synthetic_linear_model_samples(sigma, m, N)
# adversarial training - FGSM
logm0 = LogisticRegression(np.random.rand(2))


def task(k, data_k, logm0, xi, m):
    print("Starting {}".format(k))
    ret = []
    for i in range(len(xi)):
        logm_adv_fgsm, _ = adversarial_gd_fast_attack(CrossEntropy(), logm0, data_k, 1E-5, xi[i], Linf)
        logm_adv_pgd, _ = adversarial_gd_pgd_attack(CrossEntropy(), logm0, data_k, 1E-5, xi[i], Linf)
        lamb = 0.1
        logm_adv_trades, _ = adversarial_trades(CrossEntropy(), logm0, data_k, Linf, 1E-5, xi[i], lamb, int(m / 10))
        w, _ = robust_adv_data_driven_binary_classifier(xi[i], data_k)
        ret += [calculate_accuracy(data_k, lambda x: 1.0 if logm_adv_fgsm.value(x) >= 0.5 else 0.0),
                calculate_accuracy(data_k, lambda x: 1.0 if logm_adv_pgd.value(x) >= 0.5 else 0.0),
                calculate_accuracy(data_k, lambda x: 1.0 if logm_adv_trades.value(x) >= 0.5 else 0.0),
                calculate_accuracy(data_k, lambda x: 1.0 if x.dot(w) >= 0.0 else 0.0)]
    return_dict = {k: ret}
    print("Finished {}".format(k))
    return return_dict


t0 = time.time()
multiprocessing.freeze_support()
with multiprocessing.Pool(8) as pool:
    acc = pool.starmap(task, zip(range(len(data)), data, repeat(logm0), repeat(xi), repeat(m)))

time.time() - t0

columns = [a + str(b) for b in xi for a in
           ["Adv training - FGSM - xi=", "Adv training - PGD - xi=", "TRADES - xi=", "Our model - xi="]]
robustness_results = pd.DataFrame.from_dict(dict(ChainMap(*acc)), orient='index', columns=columns)

robustness_results.describe()
robustness_results.hist()
