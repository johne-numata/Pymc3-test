import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

np.random.seed(314)
N = 100
r = 0.8
x_1 = np.random.normal(size=N)
x_2 = np.random.normal(loc=x_1 * r, scale=(1 - r **2)**0.5)
y = np.random.normal(loc=x_1 - x_2)
X = np.vstack((x_1, x_2))

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

scatter_plot(X, y)
plt.savefig('img431.png')

with pm.Model() as model_ma:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10, shape=2)
	epsolon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, X)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsolon, observed=y)
	
	trace_ma = pm.sample(5000, njobs=1)

pm.traceplot(trace_ma)
plt.savefig('img432.png')
pm.summary(trace_ma[500:])
pm.autocorrplot(trace_ma[500:])
plt.savefig('img4321.png')

plt.figure()
#print(trace_red['beta'].T)
sns.kdeplot((trace_ma['beta'].T)[0], (trace_ma['beta'].T)[1])
plt.savefig('img4322.png')

pm.forestplot(trace_ma, varnames={'beta'})
plt.savefig('img4323.png')

plt.figure()
ppc = pm.sample_ppc(trace_ma, samples=2000, model=model_ma)
for y_p in ppc['y_pred']:
	sns.kdeplot(y_p, alpha=0.5, c='g')
sns.kdeplot(y, c='b')
plt.savefig('img4324.png')


