from typing import Callable
import numpy as np
import pandas as pd
from utils.loss import CrossEntropy
from utils.model import LogisticRegression
from utils.norm import Linf
from adversary.solver import adversarial_gd_fast_attack, robust_adv_data_driven_binary_classifier, \
    adversarial_gd_pgd_attack, adversarial_trades
from data.linear_data_generator import generate_synthetic_linear_model_samples
from utils.solver import gd
from utils.types import Data

# Parameters
sigma = 0.1  # noise
m = 1000  # each sample size
N = 1000  # number of samples
xi = 0.1  # adversarial power


def calculate_accuracy(data: Data, model: Callable[[np.ndarray], float]):
    x = [d.get('x') for d in data]
    y = [d.get('y') for d in data]
    correct_classification = 0
    for i in range(len(y)):
        y_model = model(x[i])
        if y_model == y[i]:
            correct_classification += 1
    return correct_classification / m


columns = ["Logistic regression", "Adv training - FGSM", "Adv training - PGD", "TRADES", "Our model"]
accuracy_results = pd.DataFrame(np.zeros([N, 5]), columns=columns)
data = generate_synthetic_linear_model_samples(sigma, m, N)

logm0 = LogisticRegression(np.random.rand(2))
for k in range(len(data)):
    data_k = data[k]
    logm, _ = gd(CrossEntropy(), logm0, data_k, 1E-5)
    accuracy_results.iloc[k, 0] = calculate_accuracy(data_k,
                                                     lambda x: 1.0 if logm.value(x) >= 0.5 else 0.0)

    logm_adv_fgsm, _ = adversarial_gd_fast_attack(CrossEntropy(), logm0, data_k, 1E-5, xi, Linf)
    accuracy_results.iloc[k, 1] = calculate_accuracy(data_k,
                                                     lambda x: 1.0 if logm_adv_fgsm.value(x) >= 0.5 else 0.0)
    logm_adv_pgd, _ = adversarial_gd_pgd_attack(CrossEntropy(), logm0, data_k, 1E-5, xi, Linf)
    accuracy_results.iloc[k, 2] = calculate_accuracy(data_k,
                                                     lambda x: 1.0 if logm_adv_pgd.value(x) >= 0.5 else 0.0)
    lamb = 0.1
    logm_adv_trades, _ = adversarial_trades(CrossEntropy(), logm0, data_k, Linf, 1E-5, xi, lamb, int(m / 10))
    accuracy_results.iloc[k, 3] = calculate_accuracy(data_k,
                                                     lambda x: 1.0 if logm_adv_trades.value(x) >= 0.5 else 0.0)
    w = robust_adv_data_driven_binary_classifier(xi, data_k)
    accuracy_results.iloc[k, 4] = calculate_accuracy(data_k,
                                                     lambda x: 1.0 if x.dot(w) >= 0.0 else 0.0)

accuracy_results.describe()
accuracy_results.hist()
