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
x_n = 'petal_length'
x_0 = df[x_n].values

with pm.Model() as model_0:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10)
	mu = alpha + pm.math.dot(x_0, beta)
	theta = pm.Deterministic('theta', 1/(1 + pm.math.exp(-mu)))
	bd = pm.Deterministic('bd', -alpha/beta)
	
	yl = pm.Bernoulli('yl', p=theta, observed=y_0)
	trace_0 = pm.sample(5000)

chain_0 = trace_0[1000:]
varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_0, varnames)
plt.savefig('img505b.png')

pm.summary(trace_0, varnames)

#print(chain_0['theta'])
plt.figure()

theta = chain_0['theta'].mean(axis=0)
idx = np.argsort(x_0)
plt.plot(x_0[idx], theta[idx], color='b', lw=3);
plt.axvline(chain_0['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(chain_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

plt.plot(x_0, y_0, 'o', color='k')
theta_hpd = pm.hpd(chain_0['theta'])[idx]
plt.fill_between(x_0[idx], theta_hpd[:, 0], theta_hpd[:, 1], color='b', alpha=0.5)
plt.xlabel(x_n, fontsize=16)
plt.ylabel(r'$\theta$', rotation=0, fontsize=16)
plt.savefig('img506b.png')



