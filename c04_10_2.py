import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

np.random.seed(314)
N = 100
alpha_real = 2.5
beta_real = [0.9, 1.5]
eps_real = np.random.normal(0, 0.5, size=N)

X = np.array([np.random.normal(i, j, N) for i,j in zip([10, 2], [1, 1.5])])
X_mean = X.mean(axis=1, keepdims=True)
X_centered = X - X_mean
y = alpha_real + np.dot(beta_real, X) + eps_real
#print(X)
#print(y)

def scatter_plot(x, y):
	plt.figure(figsize=(10,10))
	for idx, x_i in enumerate(x):
		plt.subplot(2, 2, idx + 1)
		plt.scatter(x_i, y)
		plt.xlabel('$x_{}$'.format(idx + 1), fontsize=16)
		plt.ylabel('$y$', rotation=0, fontsize=16)
	plt.subplot(2, 2, idx + 2)
	plt.scatter(x[0], x[1])
	plt.xlabel('$x_{}$'.format(idx), fontsize=16)
	plt.ylabel('$x_{}$'.format(idx + 1), rotation=0, fontsize=16)

#scatter_plot(X_centered, y)
#plt.savefig('img425.png')

with pm.Model() as model_mlr:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=1, shape=2)
	epsolon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, X)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsolon, observed=y)
	
	trace_mlr = pm.sample(5000, njobs=1)

varnames = ['alpha', 'beta', 'epsilon']
pm.traceplot(trace_mlr, varnames)
plt.savefig('img426.png')
pm.summary(trace_mlr[500:], varnames)
pm.autocorrplot(trace_mlr[500:], varnames)
plt.savefig('img4261.png')


