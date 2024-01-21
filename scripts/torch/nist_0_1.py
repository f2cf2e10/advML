import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scripts.torch.utils import train_and_analyze_models

# Using only 0s and 1s
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
zeros_ones_train = list(filter(lambda x: np.isin(x[1], [0, 1]), mnist_train)) #zero and ones

mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
zeros_ones_test = list(filter(lambda x: np.isin(x[1], [0, 1]), mnist_test)) #zero and ones

train_data = DataLoader(zeros_ones_train, batch_size=100, shuffle=False)
test_data = DataLoader(zeros_ones_test, batch_size=100, shuffle=False)

torch.manual_seed(171)
tol = 1E-5
xis = [0.05, 0.1, 0.15, 0.2]
lamb = 1.0
norm_bound = 1.0
maxIter = 10
N = 28 * 28
loss_fn = nn.BCEWithLogitsLoss()
adv_loss_fn = nn.BCEWithLogitsLoss()
model = nn.Linear(N, 1)
train = False

for xi in xis:
    train_and_analyze_models(xi, N, train, train_data, test_data, maxIter, norm_bound, model_name='nist_0_1')
