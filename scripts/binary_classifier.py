import numpy as np
import matplotlib.pyplot as plt

from utils.loss import CrossEntropy
from utils.model import LogisticRegression
from utils.norm import Linf
from utils.solver import gd
from adversary.solver import adversarial_gd_fast_attack, robust_adv_data_driven_binary_classifier, \
    adversarial_gd_pgd_attack, adversarial_trades
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_sample, \
    generate_synthetic_linear_model_with_cap_normal_distr_sample

# Parameters
# model
sigma = 0.1
n = 1000
# adversarial power
xi = 0.1

# training - logistic regression
#a, b, d, data, x, y = generate_synthetic_linear_model_with_uniform_distr_sample(sigma, n, with_const=True)
a, b, d, data, x, y = generate_synthetic_linear_model_with_cap_normal_distr_sample(sigma, n, with_const=True)
logm0 = LogisticRegression(np.random.rand(x.shape[1]))
logm, conv = gd(CrossEntropy(), logm0, data, 1E-5)
plt.figure()
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=35)
plt.scatter(x[:, 0], x[:, 1], marker="+", c=[1.0 if logm.value(x_i) >= 0.5 else 0.0 for x_i in x], s=35)
plt.title('Logistic regression')
plt.show()
plt.figure()
plt.plot(conv)
plt.title('Logistic regression - loss')
plt.show()

# adversarial training - FGSM
a, b, d, data, x, y = generate_synthetic_linear_model_with_cap_normal_distr_sample(sigma, n, with_const=True)
logm_adv_fgsm, conv_adv_fgsm = adversarial_gd_fast_attack(CrossEntropy(), logm0, data, 1E-5, xi, Linf)
plt.figure()
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=35)
plt.scatter(x[:, 0], x[:, 1], marker="+", c=[1.0 if logm_adv_fgsm.value(xi) >= 0.5 else 0.0 for xi in x], s=35)
plt.title('Adv training - FGSM')
plt.show()
plt.figure()
plt.plot(conv_adv_fgsm)
plt.title('Adv training - FGSM - loss')
plt.show()

# adversarial training - PGD
a, b, d, data, x, y = generate_synthetic_linear_model_with_cap_normal_distr_sample(sigma, n, with_const=True)
logm_adv_pgd, conv_adv_pgd = adversarial_gd_pgd_attack(CrossEntropy(), logm0, data, 1E-5, xi, Linf)
plt.figure()
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=35)
plt.scatter(x[:, 0], x[:, 1], marker="+", c=[1.0 if logm_adv_pgd.value(xi) >= 0.5 else 0.0 for xi in x], s=35)
plt.title('Adv training - PGD')
plt.show()
plt.figure()
plt.plot(conv_adv_pgd)
plt.title('Adv training - PGD - loss')
plt.show()

# adversarial training - TRADES
a, b, d, data, x, y = generate_synthetic_linear_model_with_cap_normal_distr_sample(sigma, n, with_const=True)
lamb = 0.1
logm_adv_trades, conv_adv_trades = adversarial_trades(CrossEntropy(), logm0, data, Linf, 1E-5, xi, lamb, int(n / 10))
plt.figure()
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=35)
plt.scatter(x[:, 0], x[:, 1], marker="+", c=[1.0 if logm_adv_trades.value(xi) >= 0.5 else 0.0 for xi in x], s=35)
plt.title('Adv training - Trades (lambda={:.2f})'.format(lamb))
plt.show()
plt.figure()
plt.plot(conv_adv_trades)
plt.title('Adv training - Trades (lambda={:.2f}) loss'.format(lamb))
plt.show()

# adversarial training - Our model
a, b, d, data, x, y = generate_synthetic_linear_model_with_cap_normal_distr_sample(sigma, n, with_const=True)
w, _ = robust_adv_data_driven_binary_classifier(xi, data)
plt.figure()
plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, s=35)
plt.scatter(x[:, 0], x[:, 1], marker="+", c=[1.0 if x_i.dot(w) >= 0.0 else 0.0 for x_i in x], s=35)
plt.title('Adv training - Our model')
plt.show()
