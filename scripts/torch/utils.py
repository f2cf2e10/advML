import numpy as np
import torch
from torch import nn
from adversary.torch.solver import adversarial_training_fast_gradient_sign_method, \
    adversarial_training_projected_gradient_descent, adversarial_training_trades, \
    robust_adv_data_driven_binary_classifier, robust_stable_adv_data_driven_binary_classifier
from utils.eval import RoMa
from utils.torch.solver import training


def train_and_analyze_models(xi, N, train, train_data, test_data, maxIter, norm_bound, model_name):
    print("xi={}".format(xi))
    torch.manual_seed(171)
    loss_fn = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    model = nn.Linear(N, 1)
    if not train:
        model.load_state_dict(torch.load("./models/{}_training_xi_{}.pth".format(model_name, xi)))
        model.eval()
    print(
        "Method\tTrain Acc\tTrain Loss\tPlain Test Acc\tPlain Test Loss\tFGSM Test Acc\tFGSM Test Loss\tPGD Test Acc\t" +
        "PGD Test Loss\tTRADES Test Acc\tTRADES Test Loss")
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = training(train_data, model, loss_fn, train)
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
    if train:
        torch.save(model.state_dict(), "./models/{}_training_xi_{}.pth".format(model_name, xi))

    model_robust_fgsm = nn.Linear(N, 1)
    if not train:
        model_robust_fgsm.load_state_dict(torch.load("./models/{}_fgsm_xi_{}.pth".format(model_name, xi)))
        model_robust_fgsm.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = adversarial_training_fast_gradient_sign_method(
            train_data, model_robust_fgsm, loss_fn, adv_loss_fn, train, xi=xi, norm_bound=norm_bound)
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
    if train:
        torch.save(model_robust_fgsm.state_dict(), "./models/{}_fgsm_xi_{}.pth".format(model_name, xi))

    model_robust_pgd = nn.Linear(N, 1)
    if not train:
        model_robust_pgd.load_state_dict(torch.load("./models/{}_pgd_xi_{}.pth".format(model_name, xi)))
        model_robust_pgd.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = adversarial_training_projected_gradient_descent(
            train_data, model_robust_pgd, loss_fn, adv_loss_fn, train, xi=xi, norm_bound=norm_bound)
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
    if train:
        torch.save(model_robust_pgd.state_dict(), "./models/{}_pgd_xi_{}.pth".format(model_name, xi))

    model_robust_trades = nn.Linear(N, 1)
    if not train:
        model_robust_trades.load_state_dict(torch.load("./models/{}_trades_xi_{}.pth".format(model_name, xi)))
        model_robust_trades.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.KLDivLoss()
    delta = np.Inf
    previous_train_loss = np.Inf
    for _ in range(maxIter):
        train_err, train_loss = adversarial_training_trades(
            train_data, model_robust_trades, loss_fn, adv_loss_fn, train, xi=xi)
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
    if train:
        torch.save(model_robust_trades.state_dict(), "./models/{}_trades_xi_{}.pth".format(model_name, xi))

    adv_loss_fn = nn.BCEWithLogitsLoss()
    if train:
        our_model, adv_our_err, adv_our_loss = robust_adv_data_driven_binary_classifier(train_data, xi=xi)
    else:
        our_model = nn.Linear(N, 1)
        if not train:
            our_model.load_state_dict(torch.load("./models/{}_ours_xi_{}.pth".format(model_name, xi)))
            our_model.eval()
            adv_our_err, adv_our_loss = np.NaN, np.NaN
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
    if train:
        torch.save(our_model.state_dict(), "./models/{}_ours_xi_{}.pth".format(model_name, xi))
    print()

    prob = 0.5
    n = 1000
    data_size = len(test_data.dataset)
    robustness_model = [RoMa(prob, xi, test_data.dataset[i], 1000, model) for i in range(data_size)]
    robustness_model_pgd = [RoMa(prob, xi, test_data.dataset[i], 1000, model_robust_pgd) for i in range(data_size)]
    robustness_model_fgsm = [RoMa(prob, xi, test_data.dataset[i], 1000, model_robust_fgsm) for i in range(data_size)]
    robustness_model_trades = [RoMa(prob, xi, test_data.dataset[i], 1000, model_robust_trades) for i in
                               range(data_size)]
    robustness_model_ours = [RoMa(prob, xi, test_data.dataset[i], 1000, our_model) for i in range(data_size)]

    print("Plain\t{:.7f}\t{:.7f}".format(np.mean(robustness_model), np.std(robustness_model)))
    print("FGSM\t{:.7f}\t{:.7f}".format(np.mean(robustness_model_fgsm), np.std(robustness_model_fgsm)))
    print("PGD\t{:.7f}\t{:.7f}".format(np.mean(robustness_model_pgd), np.std(robustness_model_pgd)))
    print("TRADES\t{:.7f}\t{:.7f}".format(np.mean(robustness_model_trades), np.std(robustness_model_trades)))
    print("Ours\t{:.7f}\t{:.7f}".format(np.mean(robustness_model_ours), np.std(robustness_model_ours)))
    print("======================================")


def train_and_analyze_stabilization(xi, lambs, N, train, train_data, test_data, norm_bound, model_name):
    print("xi={}".format(xi))
    torch.manual_seed(171)
    loss_fn = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    if train:
        our_model, adv_our_err, adv_our_loss = robust_adv_data_driven_binary_classifier(train_data, xi=xi)
    else:
        our_model = nn.Linear(N, 1)
        if not train:
            our_model.load_state_dict(torch.load("./models/stabilized/{}_model_xi_{}.pth".format(model_name, xi)))
            our_model.eval()
            adv_our_err, adv_our_loss = np.NaN, np.NaN
    train_err, train_loss = training(train_data, our_model, loss_fn)
    test_err, test_loss = training(test_data, our_model, loss_fn)
    adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
        test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
    adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
        test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
    adv_trades_err, adv_trades_loss = adversarial_training_trades(
        test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
    print("Model\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
        (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
        (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
    if train:
        torch.save(our_model.state_dict(), "./models/{}_model_xi_{}.pth".format(model_name, xi))
    print()

    loss_fn = nn.BCEWithLogitsLoss()
    adv_loss_fn = nn.BCEWithLogitsLoss()
    for lamb in lambs:
        print("lambda={}".format(lamb))
        if train:
            our_model, adv_our_err, adv_our_loss = robust_stable_adv_data_driven_binary_classifier(train_data, xi=xi,
                                                                                                   lamb=lamb)
        else:
            our_model = nn.Linear(N, 1)
            if not train:
                our_model.load_state_dict(
                    torch.load("./models/stabilized/{}_stable_model_xi_{}_lamb_{}.pth".format(model_name, xi, lamb)))
                our_model.eval()
                adv_our_err, adv_our_loss = np.NaN, np.NaN
        train_err, train_loss = training(train_data, our_model, loss_fn)
        test_err, test_loss = training(test_data, our_model, loss_fn)
        adv_sign_err, adv_sign_loss = adversarial_training_fast_gradient_sign_method(
            test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_pgd_err, adv_pgd_loss = adversarial_training_projected_gradient_descent(
            test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        adv_trades_err, adv_trades_loss = adversarial_training_trades(
            test_data, our_model, loss_fn, adv_loss_fn, xi=xi, norm_bound=norm_bound)
        print("Model\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}".format(
            (1 - train_err) * 100, train_loss, (1 - test_err) * 100, test_loss, (1 - adv_sign_err) * 100, adv_sign_loss,
            (1 - adv_pgd_err) * 100, adv_pgd_loss, (1 - adv_trades_err) * 100, adv_trades_loss), end='\r')
        if train:
            torch.save(our_model.state_dict(), "./models/{}_stable_model_xi_{}_lamb_{}.pth".format(model_name, xi, lamb))
        print()

    prob = 0.5
    data_size = len(test_data.dataset)
    robustness_model_Model = [RoMa(prob, xi, test_data.dataset[i], 1000, our_model) for i in range(data_size)]

    print("Model\t{:.7f}\t{:.7f}".format(np.mean(robustness_model_Model), np.std(robustness_model_Model)))
    print("======================================")
