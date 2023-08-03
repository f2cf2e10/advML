import pandas as pd
import matplotlib

matplotlib.rc('axes', titlesize=25)
matplotlib.rc('legend', fontsize=25)
matplotlib.rc('axes', labelsize=22)
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)

# 0/1
nist_0_05 = pd.read_csv("./NIST_01_0_05.csv", index_col=0, sep=";")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_05.columns = ['xi=0.0', 'xi=0.05']
nist_0_1 = pd.read_csv("./NIST_01_0_1.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_1.columns = ['xi=0.1']
nist_0_15 = pd.read_csv("./NIST_01_0_15.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_15.columns = ['xi=0.15']
nist_0_2 = pd.read_csv("./NIST_01_0_2.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_2.columns = ['xi=0.2']

nist_01 = pd.concat([nist_0_05, nist_0_1, nist_0_15, nist_0_2], join='inner', axis=1)
nist_01.index = ['Training', 'FGSM', 'Proposed Approach']
ax = nist_01.plot.bar(figsize=(12, 9), rot=0)
ax.set_ylim(0, 100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

ax = nist_01.T.plot(figsize=(12, 9), style='o--', linewidth=3, rot=0)
ax.set_ylim(0, 100)
ax.set_label('xi')
ax.set_ylabel('Accuracy [%]')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

# 3/8
nist_0_05 = pd.read_csv("./NIST_38_0_05.csv", index_col=0, sep=";")[['Plain Test Acc', 'FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_05.columns = ['xi=0.0', 'xi=0.05']
nist_0_1 = pd.read_csv("./NIST_38_0_1.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_1.columns = ['xi=0.1']
nist_0_15 = pd.read_csv("./NIST_38_0_15.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_15.columns = ['xi=0.15']
nist_0_2 = pd.read_csv("./NIST_38_0_2.csv", index_col=0, sep=";")[['FGSM Test Acc']].loc[
    ['Plain', 'FGSM', 'Ours']]
nist_0_2.columns = ['xi=0.2']

nist_38 = pd.concat([nist_0_05, nist_0_1, nist_0_15, nist_0_2], join='inner', axis=1)
nist_38.index = ['Training', 'FGSM', 'Proposed Approach']
ax = nist_38.plot.bar(figsize=(12, 9), rot=0)
ax.set_ylim(0, 100)
ax.set_label('Training methodology')
ax.set_ylabel('Accuracy [%]')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

ax = nist_38.T.plot(figsize=(12, 9), style='o--', linewidth=3, rot=0)
ax.set_ylim(0, 100)
ax.set_label('xi')
ax.set_ylabel('Accuracy [%]')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
