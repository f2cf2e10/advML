import pandas as pd
import matplotlib

matplotlib.rc('axes', titlesize=40)
matplotlib.rc('legend', fontsize=40)
matplotlib.rc('axes', labelsize=40)
matplotlib.rc('xtick', labelsize=30)
matplotlib.rc('ytick', labelsize=30)

# NIST 0/1
nist_0_05 = pd.read_csv("./NIST_01_0_05.csv", index_col=0, sep=",")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_05.columns = ['0.0', '0.05']
nist_0_1 = pd.read_csv("./NIST_01_0_1.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_1.columns = ['0.1']
nist_0_15 = pd.read_csv("./NIST_01_0_15.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_15.columns = ['0.15']
nist_0_2 = pd.read_csv("./NIST_01_0_2.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_2.columns = ['0.2']

nist_01 = pd.concat([nist_0_05, nist_0_1, nist_0_15, nist_0_2], join='inner', axis=1)
nist_01.index = ['Non-adversarial', 'FGSM', 'PGD', 'TRADES', 'Margin-based']

# NIST 3/8
nist_0_05 = pd.read_csv("./NIST_38_0_05.csv", index_col=0, sep=",")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_05.columns = ['0.0', '0.05']
nist_0_1 = pd.read_csv("./NIST_38_0_1.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_1.columns = ['0.1']
nist_0_15 = pd.read_csv("./NIST_38_0_15.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_15.columns = ['0.15']
nist_0_2 = pd.read_csv("./NIST_38_0_2.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
nist_0_2.columns = ['0.2']

nist_38 = pd.concat([nist_0_05, nist_0_1, nist_0_15, nist_0_2], join='inner', axis=1)
nist_38.index = ['Training', 'FGSM', 'PGD', 'TRADES', 'Margin-based']

# CIFAR Cat/Dog
cifar_0_05 = pd.read_csv("./CIFAR10_catdog_0_05.csv", index_col=0, sep=",")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_05.columns = ['0.0', '0.05']
cifar_0_1 = pd.read_csv("./CIFAR10_catdog_0_1.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_1.columns = ['0.1']
cifar_0_15 = pd.read_csv("./CIFAR10_catdog_0_15.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_15.columns = ['0.15']
cifar_0_2 = pd.read_csv("./CIFAR10_catdog_0_2.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_2.columns = ['0.2']

cifar_catdog = pd.concat([cifar_0_05, cifar_0_1, cifar_0_15, cifar_0_2], join='inner', axis=1)
cifar_catdog.index = ['Training', 'FGSM', 'PGD', 'TRADES', 'Margin-based']

# CIFAR airplane/dog
cifar_0_05 = pd.read_csv("./CIFAR10_airplanedog_0_05.csv", index_col=0, sep=",")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_05.columns = ['0.0', '0.05']
cifar_0_1 = pd.read_csv("./CIFAR10_airplanedog_0_1.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_1.columns = ['0.1']
cifar_0_15 = pd.read_csv("./CIFAR10_airplanedog_0_15.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_15.columns = ['0.15']
cifar_0_2 = pd.read_csv("./CIFAR10_airplanedog_0_2.csv", index_col=0, sep=",")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'PGD', 'TRADES', 'Ours']]
cifar_0_2.columns = ['0.2']

cifar_airplanedog = pd.concat([cifar_0_05, cifar_0_1, cifar_0_15, cifar_0_2], join='inner', axis=1)
cifar_airplanedog.index = ['Non-adversarial', 'FGSM', 'PGD', 'TRADES', 'Margin-based']

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16*3, 10), sharey=True)
plt.subplots_adjust(wspace=0.075, hspace=0)
nist_01.T.plot(ax=ax1, style='o--', linewidth=3, rot=0, legend=False, title="(a)")
nist_38.T.plot(ax=ax2, style='o--', linewidth=3, rot=0, legend=False, title="(b)")
cifar_airplanedog.T.plot(ax=ax3, style='o--', linewidth=3, rot=0, legend=False, title="(c)")
cifar_catdog.T.plot(ax=ax4, style='o--', linewidth=3, rot=0, legend=False, title="(d)")
ax1.set_ylim(0, 100)
ax1.set_label('xi')
ax1.set_ylabel('Accuracy [%]')
f.legend(nist_01.T.columns, frameon=False, bbox_to_anchor=(0.5, -0.05), loc='lower center',  ncol=5)
