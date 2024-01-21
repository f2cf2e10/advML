import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier
from scripts.torch.utils import train_and_analyze_models
from utils.eval import RoMa
from utils.torch.solver import training

# Using only 3s and 8s
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
threes_eights_train = list(filter(lambda x: np.isin(x[1], [3, 8]), mnist_train))  # 3s and 8s
threes_eights_train = [(x[0], 0.0 if x[1] == 3 else 1.0) for x in threes_eights_train]

mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
threes_eights_test = list(filter(lambda x: np.isin(x[1], [3, 8]), mnist_test))  # 3s and 8s
threes_eights_test = [(x[0], 0.0 if x[1] == 3 else 1.0) for x in threes_eights_test]

train_data = DataLoader(threes_eights_train, batch_size=100, shuffle=False)
test_data = DataLoader(threes_eights_test, batch_size=100, shuffle=False)

torch.manual_seed(171)
tol = 1E-6
xis = [0.05, 0.1, 0.15, 0.2]
norm_bound = 1.0
maxIter = 10
N = 28 * 28
train = False

for xi in xis:
    train_and_analyze_models(xi, N, train, train_data, test_data, maxIter, norm_bound, model_name='nist_3_8')
