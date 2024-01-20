import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier
from utils.eval import RoMa
from utils.torch.solver import training

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
loss_fn = nn.BCEWithLogitsLoss()
adv_loss_fn = nn.BCEWithLogitsLoss()
model = nn.Linear(N, 1)

for xi in xis:
    model = nn.Linear(N, 1)
    print("Method\tTrain Acc\tTrain Loss\tPlain Test Acc\tPlain Test Loss\tFGSM Test Acc\tFGSM Test Loss\tPGD Test Acc\t" +
          "PGD Test Loss\tTRADES Test Acc\tTRADES Test Loss")
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = training(train_data, model, loss_fn, True)
        test_err, test_loss = training(test_data, model, loss_fn)
        adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
            test_data, model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
            test_data, model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_trades_err, adv_trades_loss = adversarial_training_trades(
            test_data, model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        print("Plain\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
            (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
            (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
        # if np.abs(delta) <= tol:
        #    break
    print()
    torch.save(model.state_dict(), "./models/cifar10_dog_airplane_training_xi_{}.pth".format(xi))

    model_robust_fgsm = nn.Linear(N, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = adversarial_training_fast_gradient_sign_method(
            train_data, model_robust_fgsm, loss_fn, adv_loss_fn, True, xi=xi, norm_bound=norm_bound)
        test_err, test_loss = adversarial_training_fast_gradient_sign_method(
            test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
            test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
            test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_trades_err, adv_trades_loss = adversarial_training_trades(
            test_data, model_robust_fgsm, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        print("FGSM\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
            (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
            (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
        # if np.abs(delta) <= tol:
        #    break
    print()
    torch.save(model.state_dict(), "./models/cifar10_dog_airplane_fgsm_xi_{}.pth".format(xi))

    model_robust_pgd = nn.Linear(N, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = adversarial_training_projected_gradient_descent(
            train_data, model_robust_pgd, loss_fn, adv_loss_fn, True, xi=xi, norm_bound=norm_bound)
        test_err, test_loss = adversarial_training_projected_gradient_descent(
            test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
            test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
            test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_trades_err, adv_trades_loss = adversarial_training_trades(
            test_data, model_robust_pgd, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        print("PGD\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
            (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
            (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
        # if np.abs(delta) <= tol:
        #    break
    print()
    torch.save(model.state_dict(), "./models/cifar10_dog_airplane_pgd_xi_{}.pth".format(xi))

    model_robust_trades = nn.Linear(N, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.KLDivLoss()
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = adversarial_training_trades(
            train_data, model_robust_trades, loss_fn, adv_loss_fn, True, xi=xi)
        test_err, test_loss = adversarial_training_trades(
            test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
            test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
            test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_trades_err, adv_trades_loss = adversarial_training_trades(
            test_data, model_robust_trades, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        print("TRADES\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
            (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
            (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
        delta = previous_train_loss - train_loss
        previous_train_loss = train_loss
        # if np.abs(delta) <= tol:
        #    break
    print()
    torch.save(model.state_dict(), "./models/cifar10_dog_airplane_trades_xi_{}.pth".format(xi))

    our_model, adv_our_err, adv_our_loss = robust_adv_data_driven_binary_classifier(train_data, xi=xi)
    train_err, train_loss = training(train_data, our_model, loss_fn)
    test_err, test_loss = training(test_data, our_model, loss_fn)
    adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
        test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
    adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
        test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
    adv_trades_err, adv_trades_loss = adversarial_training_trades(
        test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
    print("Ours\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
        (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
        (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
    torch.save(model.state_dict(), "./models/cifar10_dog_airplane_ours_xi_{}.pth".format(xi))

    prob = 0.5
    n = 1000
    data_size = len(test_data.dataset)
    robustness_model = [RoMa(prob, xi, test_data.dataset[i], 1000, model) for i in range(data_size)]
    robustness_model_pgd = [RoMa(prob, xi, test_data.dataset[i], 1000, model_robust_pgd) for i in range(data_size)]
    robustness_model_fgsm = [RoMa(prob, xi, test_data.dataset[i], 1000, model_robust_fgsm) for i in range(data_size)]
    robustness_model_trades = [RoMa(prob, xi, test_data.dataset[i], 1000, model_robust_trades) for i in range(data_size)]
    robustness_model_ours = [RoMa(prob, xi, test_data.dataset[i], 1000, our_model) for i in range(data_size)]
