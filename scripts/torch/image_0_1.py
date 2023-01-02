import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier
from utils.torch.model import Ours
from utils.torch.solver import training
import pylab as plt

# Using only 0s and 1s
mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
zeros_ones = mnist_train.targets <= 1
mnist_train.data = mnist_train.data[zeros_ones]
mnist_train.targets = mnist_train.targets[zeros_ones] * 1.0

mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
zeros_ones = mnist_test.targets <= 1
mnist_test.data = mnist_test.data[zeros_ones]
mnist_test.targets = mnist_test.targets[zeros_ones] * 1.0

train_data = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_data = DataLoader(mnist_test, batch_size=100, shuffle=False)

torch.manual_seed(171)
tol = 1E-3
xi = 0.1

loss_fn = nn.BCEWithLogitsLoss()
adv_loss_fn = nn.BCEWithLogitsLoss(size_average=False)

model = nn.Linear(28 * 28, 1)
opt = optim.SGD(model.parameters(), lr=1e-1)
print("Method\tTrain Err\tTrain Loss\tTest Err\tTest Loss\tAdv Test Err\tAdv Test Loss")
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = training(train_data, model, loss_fn, opt)
    test_err, test_loss = training(test_data, model, loss_fn)
    adv_err, adv_loss = adversarial_training_fast_gradient_sign_method(test_data, model, loss_fn, adv_loss_fn, xi=xi)
    print("Normal\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(train_err, train_loss, test_err, test_loss,
                                                                          adv_err, adv_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss

model_robust_fgsm = nn.Linear(784, 1)
opt = optim.SGD(model_robust_fgsm.parameters(), lr=1e-1)
loss_fn = nn.BCEWithLogitsLoss()
print("\nAdv Train Err\tAdv Train Loss\tAdv Test Err\tAdv Test Loss")
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = adversarial_training_fast_gradient_sign_method(train_data, model_robust_fgsm, loss_fn,
                                                                           adv_loss_fn, opt, xi=xi)
    test_err, test_loss = adversarial_training_fast_gradient_sign_method(test_data, model_robust_fgsm, loss_fn,
                                                                         adv_loss_fn, xi=xi)
    print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(train_err, train_loss, test_err, test_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss

model_robust_pgd = nn.Linear(784, 1)
opt = optim.SGD(model_robust_pgd.parameters(), lr=1e-1)
loss_fn = nn.BCEWithLogitsLoss()
print("\nAdv Train Err\tAdv Train Loss\tAdv Test Err\tAdv Test Loss")
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = adversarial_training_projected_gradient_descent(train_data, model_robust_pgd, loss_fn,
                                                                            adv_loss_fn, opt, xi=xi)
    test_err, test_loss = adversarial_training_fast_gradient_sign_method(test_data, model_robust_pgd, loss_fn,
                                                                         adv_loss_fn, xi=xi)
    print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(train_err, train_loss, test_err, test_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss

model_robust_trades = nn.Linear(784, 1)
opt = optim.SGD(model_robust_trades.parameters(), lr=1e-1)
loss_fn = nn.BCEWithLogitsLoss()
print("\nAdv Train Err\tAdv Train Loss\tAdv Test Err\tAdv Test Loss")
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = adversarial_training_trades(train_data, model_robust_trades, loss_fn, adv_loss_fn, opt,
                                                        xi=xi, lamb=5)
    test_err, test_loss = adversarial_training_fast_gradient_sign_method(test_data, model_robust_trades, loss_fn,
                                                                         adv_loss_fn, xi=xi)
    print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(train_err, train_loss, test_err, test_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss

print("\nAdv Train Err\tAdv Train Loss\tAdv Test Err\tAdv Test Loss")
our_model, train_loss, train_err = robust_adv_data_driven_binary_classifier(train_data.dataset, add_const=True, xi=xi)
test_err, test_loss = adversarial_training_fast_gradient_sign_method(test_data, our_model, loss_fn, adv_loss_fn, xi=xi)
print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(train_err, train_loss, test_err, test_loss))
