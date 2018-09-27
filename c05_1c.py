import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

iris = sns.load_dataset('iris', data_home="./BAP")
#print(iris.head())
"""
sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True)
plt.savefig('img503.png')

sns.pairplot(iris, hue='species', diag_kind='kde')
plt.savefig('img504.png')

"""

df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values

with pm.Model() as model_0:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10)
	mu = alpha + pm.math.dot(x_0, beta)
	epsilon = pm.HalfCauchy('epsilon', 5)
#	theta = pm.Deterministic('theta', 1/(1 + pm.math.exp(-mu)))
#	bd = pm.Deterministic('bd', -alpha/beta)
	
#	yl = pm.Bernoulli('yl', p=theta, observed=y_0)
	y = pm.Normal('y', mu=mu, sd=epsilon, observed=y_0)
	trace_0 = pm.sample(5000)

chain_0 = trace_0[1000:]
#varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_0)
plt.savefig('img505c.png')

pm.summary(trace_0)

plt.figure()
plt.plot(x_0, y_0, 'o', color='k')

y_pred = pm.sample_ppc(chain_0, 100, model_0)
y = y_pred['y'].mean(0)
#print(y)
idx = np.argsort(x_0)
plt.plot(x_0[idx], y[idx], color='r')
#theta_hpd = pm.hpd(chain_0['theta'])[idx]
#plt.fill_between(x_0[idx], theta_hpd[:, 0], theta_hpd[:, 1], color='b', alpha=0.5)
plt.xlabel(x_n, fontsize=16)
plt.ylabel('species', rotation=90, fontsize=16)
plt.savefig('img506c.png')


