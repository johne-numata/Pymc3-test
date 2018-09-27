import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

np.random.seed(1)
x = np.random.uniform(0, 10, size = 20)
y = np.random.normal(np.sin(x), 0.2)
"""
plt.plot(x, y, 'o')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16, rotation=0)
plt.savefig('img801.png')
"""

def gauss_kernel(x, n_knots):
	knots = np.linspace(x.min(), x.max(), n_knots)
	w = 2
	return np.array([np.exp(-(x-k)**2/w) for k in knots])


n_knots = 5
with pm.Model() as kernel_model:
	gamma = pm.Cauchy('gamma', alpha=0, beta=1, shape=n_knots)
	sd = pm.Uniform('sd', 0, 10)
	mu = pm.math.dot(gamma, gauss_kernel(x, n_knots))
	yl = pm.Normal('yl', mu=mu, sd=sd, observed=y)
	
	kernel_trace = pm.sample(5000, njobs=1)

#pm.traceplot(kernel_trace)
#plt.savefig('img802.png')

ppc = pm.sample_ppc(kernel_trace, model=kernel_model, samples=100)
plt.plot(x, ppc['yl'].T, 'ro', alpha=0.1)

plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('f(x)', fontsize=16, rotation=0)
plt.savefig('img803.png')

plt.figure()
new_x = np.linspace(x.min(), x.max(), 100)
k = gauss_kernel(new_x, n_knots)
gamma_pred = kernel_trace['gamma']
for i in range(100):
	idx = np.random.randint(0, len(gamma_pred))
	y_pred = np.dot(gamma_pred[idx], k)
	plt.plot(new_x, y_pred, 'r-', alpha=0.1)
plt.plot(x, y, 'bo')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)', fontsize=16, rotation=0)
plt.savefig('img804.png')


