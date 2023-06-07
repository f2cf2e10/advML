import seaborn as sns

acc_0_1 = pd.read_csv('./accuracy_torch_1000_paths_xi_0.1.csv', index_col=0).assign(xi=0.1)
acc_0_2 = pd.read_csv('./accuracy_torch_1000_paths_xi_0.2.csv', index_col=0).assign(xi=0.2)
acc_0_3 = pd.read_csv('./accuracy_torch_1000_paths_xi_0.3.csv', index_col=0).assign(xi=0.3)
acc = pd.concat([acc_0_1, acc_0_2, acc_0_3])
acc_df = pd.melt(acc, id_vars=['xi'], var_name='Method')
acc_df.columns = ['Xi', 'Method', 'Accuracy']
matplotlib.rc('axes', titlesize=22)
matplotlib.rc('axes', labelsize=22)
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
figure(figsize=(12,9))
ax = sns.boxplot(x="Xi", y="Accuracy", hue="Method", data=acc_df)
plt.show()

acc_0_1 = pd.read_csv('./accuracy_torch_1000_paths_xi_0.1.csv', index_col=0)
acc_0_2 = pd.read_csv('./accuracy_torch_1000_paths_xi_0.2.csv', index_col=0)
acc_0_3 = pd.read_csv('./accuracy_torch_1000_paths_xi_0.3.csv', index_col=0)
acc_mean = pd.concat([acc_0_1.mean(), acc_0_2.mean(), acc_0_3.mean()], axis=1)
acc_mean.columns=[0.1, 0.2, 0.3]
ax = acc_mean.T.plot(figsize=(12,9))
ax.set_xticks(acc_mean.columns)
ax.set_xlabel("Xi")
ax.set_ylabel("Accuracy")
plt.show()
