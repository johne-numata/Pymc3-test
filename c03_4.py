import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips', data_home="./BAP")
#print(tips.tail())
sns.violinplot(x='day', y='tip', data=tips)
plt.savefig('img310.png')

data = tips['tip'].values
idx = pd.Categorical(tips['day']).codes
#print(data)
#print(idx)

with pm.Model() as comparing_groups:
	means = pm.Normal('means', mu=0, sd=10, shape=len(set(idx)))
	sds = pm.HalfNormal('sds', sd=10, shape=len(set(idx)))
#	y = pm.StudentT('y', mu=means[idx], sd=sds[idx], observed=data)
	nu = pm.Exponential('nu', 1/10, shape=len(set(idx)))
	y = pm.StudentT('y', mu=means[idx], sd=sds[idx], nu=nu[idx], observed=data)
	trace_cg = pm.sample(5000)

chain_cg = trace_cg[100:]
pm.traceplot(chain_cg)
plt.savefig('img311.png')

pm.summary(chain_cg)

plt.clf()
plt.style.use('seaborn-darkgrid')

#y_pred = pm.sample_ppc(chain_cg, 100, comparing_groups)
#print(data)
#print(y_pred['y'])
#sns.kdeplot(data, color='b')
#for i in y_pred['y']:
#	sns.kdeplot(i, color='r', alpha=0.1)
#plt.xlim(35, 75)
#plt.title("Student's t model", fontsize=16)
#plt.xlabel('$x$', fontsize=16)
#plt.savefig('img312.png')

dist = stats.norm()
_, ax = plt.subplots(3, 2, figsize=(16, 12))
comparisons = [(i, j) for i in range(4) for j in range(i+1, 4)]
pos = [(k, l) for k in range(3) for l in (0, 1)]

for (i, j), (k, l) in zip(comparisons, pos):
	means_diff = chain_cg['means'][:, i] - chain_cg['means'][:, j]
	d_cohen = (means_diff / np.sqrt((chain_cg['sds'][:, i]**2 + chain_cg['sds'][:, j]**2) / 2)).mean()
	ps = dist.cdf(d_cohen / (2**0.5))

# KDEプロットを表示の場合
#	pm.plot_posterior(means_diff, ref_val=0, ax=ax[k, l], color='skyblue', kde_plot=True)
# ヒストグラムを表示の場合
	pm.plot_posterior(means_diff, ref_val=0, ax=ax[k, l], color='skyblue')
	ax[k, l].plot(0, label="Cohen's d = {: .2f}\nProb sup = {: .2f}".format(d_cohen, ps), alpha=0)
	ax[k, l].set_xlabel('$\mu_{}-\mu_{}$'.format(i, j), fontsize=18)
	ax[k, l].legend(loc=0, fontsize=14)
plt.savefig('img312.png')




