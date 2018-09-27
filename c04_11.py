import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

np.random.seed(314)
N = 100
x_1 = np.random.normal(size=N)
x_2 = x_1 + np.random.normal(size=N, scale=1)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2))
#X = x_2

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

#scatter_plot(X, y)
#plt.savefig('img427.png')

with pm.Model() as model_red:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10, shape=2)
#	beta = pm.Normal('beta', mu=0, sd=10)
	epsolon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, X)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsolon, observed=y)
	
	trace_red = pm.sample(5000, njobs=1)

varnames = ['alpha', 'beta', 'epsilon']
pm.traceplot(trace_red, varnames)
plt.savefig('img428.png')
pm.summary(trace_red[500:], varnames)
pm.autocorrplot(trace_red[500:], varnames)
plt.savefig('img4281.png')


