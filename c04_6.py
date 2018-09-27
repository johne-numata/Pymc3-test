import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

ans = sns.load_dataset('anscombe', data_home="./BAP")
x_3 = ans[ans.dataset == 'III']['x'].values
y_3 = ans[ans.dataset == 'III']['y'].values

"""
plt.subplot(1, 2, 1)
beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
plt.plot(x_3, (alpha_c + beta_c * x_3), 'k', label='y = {:.2f} + {:2f} * x'.format(alpha_c, beta_c))
plt.plot(x_3, y_3, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=0, fontsize=14)
plt.subplot(1, 2, 2)
sns.kdeplot(y_3)
plt.xlabel('$y$', fontsize=16)
plt.savefig('img414.png')
"""

with pm.Model() as model_t:
	alpha = pm.Normal('alpha', mu=0, sd=100)
	beta = pm.Normal('beta', mu=0, sd=1)
	epsilon = pm.HalfCauchy('epsilon', 5)
	nu = pm.Deterministic('nu', pm.Exponential('nu_', 1/29) + 1)
	
	y_pred = pm.StudentT('y_pred', mu=alpha + beta * x_3, sd=epsilon, nu=nu, observed= y_3)
	
	start = pm.find_MAP()
	step = pm.NUTS(scaling= start)
	trace_t = pm.sample(2000, setp = step, start = start, njobs=1)


pm.traceplot(trace_t)
plt.savefig('img4142.png')
pm.autocorrplot(trace_t)
plt.savefig('img4143.png')


plt.clf()

beta_c, alpha_c = stats.linregress(x_3, y_3)[:2]
plt.plot(x_3, (alpha_c + beta_c * x_3), 'k', label='non-robust', alpha=0.5)
plt.plot(x_3, y_3, 'bo')
alpha_m = trace_t['alpha'].mean(0)
beta_m = trace_t['beta'].mean(0)
plt.plot(x_3, alpha_m + beta_m * x_3, c='k', label='robust')

plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc=2, fontsize=14)
plt.savefig('img415.png')

plt.clf()

ppc = pm.sample_ppc(trace_t, samples=200, model=model_t, random_seed=2)
for y_tilde in ppc['y_pred']:
	sns.kdeplot(y_tilde, alpha=0.5, c='g')
sns.kdeplot(y_3, linewidth=3)
plt.savefig('img416.png')




