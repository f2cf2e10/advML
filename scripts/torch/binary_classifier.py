import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pylab as plt

from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier
from data.linear_data_generator import generate_synthetic_linear_model_with_uniform_distr_sample, \
    generate_synthetic_linear_model_with_cap_normal_distr_sample
from utils.torch.solver import training


def scatter_plot(model, title, data_x, data_y):
    plt.figure()
    plt.scatter(data_x[:, 0], data_x[:, 1], marker="o", c=data_y, s=35)
    plt.scatter(data_x[:, 0], data_x[:, 1], marker="+",
                c=[+1.0 if model(torch.Tensor(x_i)) > 0 else -1.0 for x_i in data_x], s=35)
    plt.title(title)
    plt.show()


# Parameters
# model
sigma = 0.3
tol = 1E-5
n = 1500
# adversarial power
xi = 0.1
a, b, d, data, x, y = generate_synthetic_linear_model_with_uniform_distr_sample(0.0, n, 0.75, 0.1)

train_x = torch.Tensor(x[0:1000])
train_y = (torch.Tensor(y[0:1000]) + 1) / 2
train_set = TensorDataset(train_x, train_y)
train_data = DataLoader(train_set, batch_size=100, shuffle=True)

test_x = torch.Tensor(x[1000:])
test_y = (torch.Tensor(y[1000:]) + 1) / 2
test_set = TensorDataset(test_x, test_y)
test_data = DataLoader(test_set, batch_size=100, shuffle=False)

torch.manual_seed(171)

loss_fn = nn.BCEWithLogitsLoss()
adv_loss_fn = nn.BCEWithLogitsLoss()

model = nn.Linear(d, 1)
print("Method\tTrain Acc\tTrain Loss\tPlain Test Acc\tPlain Test Loss\tFGSM Test Acc\tFGSM Test Loss\tPGD Test Acc\t" +
      "PGD Test Loss\tTRADES Test Acc\tTRADES Test Loss")
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = training(train_data, model, loss_fn, True)
    test_err, test_loss = training(test_data, model, loss_fn)
    adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
        test_data, model, loss_fn, adv_loss_fn, xi=xi)
    adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
        test_data, model, loss_fn, adv_loss_fn, xi=xi)
    adv_trades_err, adv_trades_loss = adversarial_training_trades(
        test_data, model, loss_fn, adv_loss_fn, xi=xi)
    print("Plain\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
        (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
        (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss
print()
scatter_plot(model, 'Model - train', train_x, train_y)
scatter_plot(model, 'Model - test', test_x, test_y)

model_robust_fgsm = nn.Linear(d, 1)
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = adversarial_training_fast_gradient_sign_method(
        train_data, model_robust_fgsm, loss_fn, adv_loss_fn, True, xi=xi)
    test_err, test_loss = training(test_data, model_robust_fgsm, loss_fn)
    adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
        test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi)
    adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
        test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi)
    adv_trades_err, adv_trades_loss = adversarial_training_trades(
        test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi)
    print("FGSM\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
        (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
        (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss
print()
scatter_plot(model_robust_fgsm, 'FGSM - train', train_x, train_y)
scatter_plot(model_robust_fgsm, 'FGSM - test', test_x, test_y)

model_robust_pgd = nn.Linear(d, 1)
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = adversarial_training_projected_gradient_descent(
        train_data, model_robust_pgd, loss_fn, adv_loss_fn, True, xi=xi)
    test_err, test_loss = training(test_data, model_robust_pgd, loss_fn)
    adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
        test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi)
    adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
        test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi)
    adv_trades_err, adv_trades_loss = adversarial_training_trades(
        test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi)
    print("PGD\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
        (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
        (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss
print()
scatter_plot(model_robust_pgd, 'PGD - train', train_x, train_y)
scatter_plot(model_robust_pgd, 'PGD - test', test_x, test_y)

model_robust_trades = nn.Linear(d, 1)
delta = np.Inf
previous_train_loss = np.Inf
while delta > tol:
    train_err, train_loss = adversarial_training_trades(
        train_data, model_robust_trades, loss_fn, adv_loss_fn, True, xi=xi)
    test_err, test_loss = training(test_data, model_robust_trades, loss_fn)
    adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
        test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi)
    adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
        test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi)
    adv_trades_err, adv_trades_loss = adversarial_training_trades(
        test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi)
    print("TRADES\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
        (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
        (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
    delta = previous_train_loss - train_loss
    previous_train_loss = train_loss
print()
scatter_plot(model_robust_trades, 'TRADES - train', train_x, train_y)
scatter_plot(model_robust_trades, 'TRADES - test', test_x, test_y)

our_model, adv_our_err, adv_our_loss = robust_adv_data_driven_binary_classifier(train_data, xi=xi)
test_err, test_loss = training(test_data, our_model, loss_fn)
adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
    test_data, our_model, loss_fn, adv_loss_fn, xi=xi)
adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
    test_data, our_model, loss_fn, adv_loss_fn, xi=xi)
adv_trades_err, adv_trades_loss = adversarial_training_trades(
    test_data, our_model, loss_fn, adv_loss_fn, xi=xi)
print("Ours\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
    (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
    (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')

scatter_plot(our_model, 'Ours - train', train_x, train_y)
scatter_plot(our_model, 'Ours - test', test_x, test_y)
