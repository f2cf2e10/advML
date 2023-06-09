import pandas as pd
import matplotlib

matplotlib.rc('axes', titlesize=22)
matplotlib.rc('legend', fontsize=14)
matplotlib.rc('axes', labelsize=22)
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)

#cat/dog
nist_0_01 = pd.read_csv("./CIFAR10_0_01_catdog.csv", index_col=0, sep=";")[['Plain Test Acc', 'FGSM Test Acc']].loc[['Plain', 'FGSM', 'TRADES', 'Ours']]
nist_0_01.columns = ['Genuine', 'Adversarial xi=0.01']
nist_0_05 = pd.read_csv("./CIFAR10_0_05_catdog.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[['Plain', 'FGSM', 'TRADES', 'Ours']]
nist_0_05.columns = ['Adversarial xi=0.05']
nist_0_1 = pd.read_csv("./CIFAR10_0_1_catdog.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[['Plain', 'FGSM', 'TRADES', 'Ours']]
nist_0_1.columns = ['Adversarial xi=0.1']

nist_01 = pd.concat([nist_0_01, nist_0_05, nist_0_1], join='inner', axis=1)
nist_01.index = ['Training', 'FGSM', 'TRADES', 'Ours']
ax = nist_01.plot.bar(figsize=(12,9), rot=0)
ax.set_ylim(0,100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)


#airplane/dog
nist_0_01 = pd.read_csv("./CIFAR10_0_01_airplanedog.csv", index_col=0, sep=";")[['Plain Test Acc', 'FGSM Test Acc']].loc[['Plain', 'FGSM', 'TRADES', 'Ours']]
nist_0_01.columns = ['Genuine', 'Adversarial xi=0.01']
nist_0_05 = pd.read_csv("./CIFAR10_0_05_airplanedog.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[['Plain', 'FGSM', 'TRADES', 'Ours']]
nist_0_05.columns = ['Adversarial xi=0.05']
nist_0_1 = pd.read_csv("./CIFAR10_0_1_airplanedog.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[['Plain', 'FGSM', 'TRADES', 'Ours']]
nist_0_1.columns = ['Adversarial xi=0.1']

nist_38 = pd.concat([nist_0_01, nist_0_05, nist_0_1], join='inner', axis=1)
nist_38.index = ['Training', 'FGSM', 'TRADES', 'Ours']
ax = nist_38.plot.bar(figsize=(12,9), rot=0)
ax.set_ylim(0,100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
