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
N = 1000  # number of samples
xi = [0.01, 0.1, 1.0]  # adversarial power


def calculate_accuracy(data: Data, model: Callable[[np.ndarray], float]):
    x = [d.get('x') for d in data]
    y = [d.get('y') for d in data]
    correct_classification = 0
    for i in range(len(y)):
        y_model = model(x[i])
        if y_model == y[i]:
            correct_classification += 1
    return correct_classification / m


columns = [a + str(b) for b in xi for a in
           ["Adv training - FGSM - xi=", "Adv training - PGD - xi=", "TRADES - xi=", "Our model - xi="]]
robustness_results = pd.DataFrame(np.zeros([N, 1 + 3 * len(xi)]), columns=columns)
data = generate_synthetic_linear_model_samples(sigma, m, N)
# adversarial training - FGSM
logm0 = LogisticRegression(np.random.rand(2))
for k in range(len(data)):
    data_k = data[k]
    for i in range(len(xi)):
        logm_adv_fgsm, _ = adversarial_gd_fast_attack(CrossEntropy(), logm0, data_k, 1E-5, xi[i], Linf)
        robustness_results.iloc[k, 0 + len(xi) * i] = calculate_accuracy(data_k,
                                                                         lambda x: 1.0 if logm_adv_fgsm.value(
                                                                             x) >= 0.5 else 0.0)
        logm_adv_pgd, _ = adversarial_gd_pgd_attack(CrossEntropy(), logm0, data_k, 1E-5, xi[i], Linf)
        robustness_results.iloc[k, 1 + len(xi) * i] = calculate_accuracy(data_k,
                                                                         lambda x: 1.0 if logm_adv_pgd.value(
                                                                             x) >= 0.5 else 0.0)
        lamb = 0.1
        logm_adv_trades, _ = adversarial_trades(CrossEntropy(), logm0, data_k, Linf, 1E-5, xi[i], lamb, int(m / 10))
        robustness_results.iloc[k, 2 + len(xi) * i] = calculate_accuracy(data_k,
                                                                         lambda x: 1.0 if logm_adv_trades.value(
                                                                             x) >= 0.5 else 0.0)
        w = robust_adv_data_driven_binary_classifier(xi[i], data_k)
        robustness_results.iloc[k, 3 + len(xi) * i] = calculate_accuracy(data_k,
                                                                         lambda x: 1.0 if x.dot(w) >= 0.0 else 0.0)

robustness_results.describe()
robustness_results.hist()
