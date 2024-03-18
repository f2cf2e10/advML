import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scripts.torch.utils import train_and_analyze_models

# Using only airplane (0) and dog(5)
cifar10_train = datasets.CIFAR10("../data", train=True, download=True, transform=transforms.ToTensor())
airplane_dog_train = list(filter(lambda x: np.isin(x[1], [0, 5]), cifar10_train)) #airplanes and dogs
airplane_dog_train = [(x[0], 0.0 if x[1] == 0 else 1.0) for x in airplane_dog_train]

cifar10_test = datasets.CIFAR10("../data", train=False, download=True, transform=transforms.ToTensor())
airplane_dog_test = list(filter(lambda x: np.isin(x[1], [0, 5]), cifar10_test)) #airplanes and dogs
airplane_dog_test = [(x[0], 0.0 if x[1] == 0 else 1.0) for x in airplane_dog_test]

train_data = DataLoader(airplane_dog_train, batch_size=200, shuffle=False)
test_data = DataLoader(airplane_dog_test, batch_size=200, shuffle=False)

torch.manual_seed(171)
tol = 1E-5
xis = [0.05, 0.1, 0.15, 0.2]
lamb = 1.0
norm_bound = 1.0
maxIter = 10
N = 3 * 32 * 32
train = False

for xi in xis:
    train_and_analyze_models(xi, N, train, train_data, test_data, maxIter, norm_bound, model_name='cifar10_dog_airplane')
