import matplotlib
import pandas as pd

matplotlib.rc('axes', titlesize=22)
matplotlib.rc('legend', fontsize=14)
matplotlib.rc('axes', labelsize=22)
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)

# cat/dog
cifar_0_05 = pd.read_csv("./CIFAR10_catdog_0_05.csv", index_col=0, sep=";")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'TRADES', 'Ours']]
cifar_0_05.columns = ['Genuine', 'Adversarial xi=0.05']
cifar_0_1 = pd.read_csv("./CIFAR10_catdog_0_1.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'TRADES', 'Ours']]
cifar_0_1.columns = ['Adversarial xi=0.1']
cifar_0_25 = pd.read_csv("./CIFAR10_catdog_0_25.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'TRADES', 'Ours']]
cifar_0_25.columns = ['Adversarial xi=0.25']

cifar_catdog = pd.concat([cifar_0_05, cifar_0_1, cifar_0_25], join='inner', axis=1)
cifar_catdog.index = ['Training', 'FGSM', 'TRADES', 'Ours']
ax = cifar_catdog.plot.bar(figsize=(12, 9), rot=0)
ax.set_ylim(0, 100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

# airplane/dog
cifar_0_05 = pd.read_csv("./CIFAR10_airplanedog_0_05.csv", index_col=0, sep=";")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'TRADES', 'Ours']]
cifar_0_05.columns = ['Genuine', 'Adversarial xi=0.05']
cifar_0_1 = pd.read_csv("./CIFAR10_airplanedog_0_1_.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'TRADES', 'Ours']]
cifar_0_1.columns = ['Adversarial xi=0.1']
cifar_0_25 = pd.read_csv("./CIFAR10_airplanedog_0_25.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'TRADES', 'Ours']]
cifar_0_25.columns = ['Adversarial xi=0.25']

cifar_ariplanedog = pd.concat([cifar_0_05, cifar_0_1, cifar_0_25], join='inner', axis=1)
cifar_ariplanedog.index = ['Training', 'FGSM', 'TRADES', 'Ours']
ax = cifar_ariplanedog.plot.bar(figsize=(12, 9), rot=0)
ax.set_ylim(0, 100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

