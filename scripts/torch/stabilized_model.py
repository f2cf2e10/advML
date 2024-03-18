import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scripts.torch.utils import train_and_analyze_stabilization

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
lambs = [0.1, 1, 10]

for xi in xis:
    train_and_analyze_stabilization(xi, lambs, N, train, train_data, test_data, maxIter, norm_bound, model_name='nist_3_8')