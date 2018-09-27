import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

N = 20
M = 8
idx = np.repeat(range(M-1), N)
#print(idx)
idx = np.append(idx, 7)
#print(idx)
np.random.seed(314)

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real

plt.figure(figsize=(16, 8))
j, k = 0, N
for i in range(M):
	plt.subplot(2, 4, i+1)
	plt.scatter(x_m[j:k], y_m[j:k])
	plt.xlabel('$x_{}$'.format(i), fontsize=16)
	plt.ylabel('$y$', fontsize=16, rotation=0)
	plt.xlim(6, 15)
	plt.ylim(7, 17)
	j += N
	k += N

plt.tight_layout()
plt.savefig('img417.png')

x_centered = x_m - x_m.mean()

with pm.Model() as unpooled_model:
	alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=10, shape=M)
	beta = pm.Normal('beta', mu=0, sd=10, shape=M)
	epsilon = pm.HalfCauchy('epsilon', 5)
	nu = pm.Exponential('nu', 1/30)
	ypred = pm.StudentT('y_pred', mu=alpha_tmp[idx] + beta[idx] * x_centered, sd=epsilon, 
	nu=nu, observed=y_m)
	alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_m.mean())
	
	start = pm.find_MAP()
	step = pm.NUTS(scaling=start)
	trace_up = pm.sample(2000, step=step, start=start, njobs=1)

varnames = ['alpha', 'beta', 'epsilon', 'nu']
pm.traceplot(trace_up, varnames)
plt.savefig('img418.png')





