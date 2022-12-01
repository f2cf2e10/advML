import numpy as np
import matplotlib.pyplot as plt

from utils.loss import L2
from utils.model import LinearRegression
from utils.norm import Linf
from utils.solver import gd
from adversary.solver import adversarial_gd_fast_attack

# Random cloud of points
a = 2.0
b = 0.0
np.random.seed(17171717)
xi = 0.1

## Linear model
x = np.arange(-1, 1, 0.01)
y = a * x + b + np.random.randn(len(x))
data = [{'x': np.array([x[i], 1.]), 'y': y[i]} for i in range(len(x))]
lm0 = LinearRegression(np.random.randn(2))
lm, conv = gd(L2(), lm0, data, 1E-7)
lm_adv, conv_adv = adversarial_gd_fast_attack(L2(), lm0, data, 1E-7, xi, Linf)
# alm, blm = lm.get_theta()
# alm_adv, blm_adv = lm_adv.get_theta()
alm = lm.get_theta()
alm_adv = lm_adv.get_theta()
blm = blm_adv = 0

plt.errorbar(x, y, xerr=xi, fmt='.', color='blue', ecolor='lightblue', capsize=5)
plt.plot(x, alm * x + blm, 'g')
plt.plot(x, alm_adv * x + blm_adv, 'r')
plt.show()

# Deterministic points
x = np.array([-1., 0.0, 1.])
y = np.array([-1., 0.0, 1.])
data = [{'x': np.array([x[i]]), 'y': y[i]} for i in range(len(x))]
lm0 = LinearRegression(np.array([1.5]))
lm, conv = gd(L2(), lm0, data, 1E-7)
lm_adv, conv_adv = adversarial_gd_fast_attack(L2(), lm0, data, 1E-7, xi, Linf)
plt.errorbar(x, y, xerr=xi, fmt='.', color='blue', ecolor='lightblue', capsize=5)
plt.plot(x, lm.get_theta() * x, 'g')
plt.plot(x, lm_adv.get_theta() * x, 'r')
plt.show()

