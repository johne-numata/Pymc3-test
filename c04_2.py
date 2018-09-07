import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

"""
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b.')
plt.xlabel('$x$', fontsize= 16)
plt.ylabel('$y$', fontsize=16, rotation= 0)
plt.plot(x, y_real, 'k')
plt.subplot(1, 2, 2)
sns.kdeplot(y)
plt.xlabel('$x$', fontsize=16)
plt.savefig('img403.png')
"""

with pm.Model() as model:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10)
	epsilon = pm.HalfCauchy('epsilon', 5)
	
	mu = pm.Deterministic('mu', alpha + beta * x)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
	
	start = pm.find_MAP()
	step = pm.Metropolis()
	trace = pm.sample(11000, step, start, njobs=1)

trace_n = trace[1000:]
pm.traceplot(trace_n)
plt.savefig('img404.png')

"""
varnames = ['alpha', 'beta', 'epsilon']
pm.autocorrplot(trace_n, varnames)
plt.savefig('img405.png')

plt.clf()
sns.kdeplot(trace_n['alpha'], trace_n['beta'])
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$\beta$', fontsize=16, rotation=0)
plt.savefig('img406.png')

"""

plt.clf()
plt.plot(x, y, 'b.');
alpha_m = trace_n['alpha'].mean()
beta_m = trace_n['beta'].mean()

"""

plt.plot(x, alpha_m + beta_m * x, c='k', label='y ={:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc= 2, fontsize=14)
plt.savefig('img407.png')

plt.clf()
plt.plot(x, y, 'b.');
idx = range(0, len(trace_n['alpha']), 10)
#print(np.shape(x))
#print(np.shape(trace_n['alpha'][idx] + trace_n['beta'][idx] * x[:,np.newaxis]))
plt.plot(x, trace_n['alpha'][idx] + trace_n['beta'][idx] * x[:,np.newaxis], c='gray', alpha=0.5);
plt.plot(x, alpha_m + beta_m * x, c='k', label='y ={:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc= 2, fontsize=14)
plt.savefig('img408.png')

plt.clf()
plt.plot(x, alpha_m + beta_m * x, c='k', label='y ={:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
idx = np.argsort(x)
x_ord = x[idx]
sig = pm.hpd(trace_n['mu'], alpha=0.02)[idx]
plt.fill_between(x_ord, sig[:, 0], sig[:, 1], color='gray')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img409.png')

"""

ppc = pm.sample_ppc(trace_n, samples=100000, model=model)

idx = np.argsort(x)
x_ord = x[idx]
#plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k', label= 'y = {:.2f} + {:.2f} * x'.format(alpha_m, beta_m))
sig0 = pm.hpd(ppc['y_pred'], alpha=0.5)[idx]
sig1 = pm.hpd(ppc['y_pred'], alpha=0.05)[idx]
plt.fill_between(x_ord, sig0[:, 0], sig0[:, 1], color='gray', alpha=1)
plt.fill_between(x_ord, sig1[:, 0], sig1[:, 1], color='gray', alpha=0.5)
plt.xlabel('$x$', fontsize= 16)
plt.ylabel('$y$', fontsize= 16, rotation=0)
plt.savefig('img410.png')

