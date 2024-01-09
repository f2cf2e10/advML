import matplotlib
import pandas as pd

matplotlib.rc('axes', titlesize=20)
matplotlib.rc('legend', fontsize=22)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
# cat/dog
cifar_0_05 = pd.read_csv("./CIFAR10_catdog_0_05.csv", index_col=0, sep=",")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_05.columns = ['xi=0.0', 'xi=0.05']
cifar_0_1 = pd.read_csv("./CIFAR10_catdog_0_1.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_1.columns = ['xi=0.1']
cifar_0_15 = pd.read_csv("./CIFAR10_catdog_0_15.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_15.columns = ['xi=0.1']
cifar_0_2 = pd.read_csv("./CIFAR10_catdog_0_2.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_2.columns = ['xi=0.2']

cifar_catdog = pd.concat([cifar_0_05, cifar_0_1, cifar_0_15, cifar_0_2], join='inner', axis=1)
cifar_catdog.index = ['Training', 'FGSM', 'PGD', 'TRADES', 'Our Approach']
ax = cifar_catdog.plot.bar(figsize=(12, 9), rot=0)
ax.set_ylim(0, 100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

ax = cifar_catdog.T.plot(figsize=(16, 10), style='o--', linewidth=3, rot=0)
ax.set_ylim(0, 100)
ax.set_label('xi')
ax.set_ylabel('Accuracy [%]')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)

# airplane/dog
cifar_0_05 = pd.read_csv("./CIFAR10_airplanedog_0_05.csv", index_col=0, sep=",")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_05.columns = ['xi=0.0', 'xi=0.05']
cifar_0_1 = pd.read_csv("./CIFAR10_airplanedog_0_1.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_1.columns = ['xi=0.1']
cifar_0_15 = pd.read_csv("./CIFAR10_airplanedog_0_15.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_15.columns = ['xi=0.1']
cifar_0_2 = pd.read_csv("./CIFAR10_airplanedog_0_2.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_2.columns = ['xi=0.25']

cifar_airplanedog = pd.concat([cifar_0_05, cifar_0_1, cifar_0_15, cifar_0_2], join='inner', axis=1)
cifar_airplanedog.index = ['Training', 'FGSM', 'PGD', 'TRADES', 'Our Approach']
ax = cifar_airplanedog.plot.bar(figsize=(12, 9), rot=0)
ax.set_ylim(0, 100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

ax = cifar_airplanedog.T.plot(figsize=(16, 10), style='o--', linewidth=3, rot=0)
ax.set_ylim(0, 100)
ax.set_label('xi')
ax.set_ylabel('Accuracy [%]')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
