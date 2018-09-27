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
y_1 = pd.Categorical(df['species']).codes
x_n = ['petal_length', 'petal_width']
x_1 = df[x_n].values

with pm.Model() as model_1:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.StudentT('beta', mu=0, nu=3, sd=1, shape=len(x_n))
#	beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))
	mu = alpha + pm.math.dot(x_1, beta)
	theta = pm.Deterministic('theta', 1/(1 + pm.math.exp(-mu)))
	bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:, 0])
	
	yl = pm.Bernoulli('yl', p=theta, observed=y_1)
	trace_1 = pm.sample(5000)

chain_1 = trace_1[1000:]
varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_1, varnames)
plt.savefig('img507b.png')

pm.summary(trace_1, varnames)


plt.figure()

idx = np.argsort(x_1[:, 0])
ld = chain_1['bd'].mean(0)[idx]

plt.plot(x_1[:, 0][idx], ld, color='r')
plt.scatter(x_1[:, 0], x_1[:, 1], c=y_1, cmap='viridis')

ld_hpd = pm.hpd(chain_1['bd'], 0.2)[idx]
plt.fill_between(x_1[:, 0][idx], ld_hpd[:, 0], ld_hpd[:, 1], color='r', alpha=0.5)
plt.xlabel(x_n[0], fontsize=16)
plt.ylabel(x_n[1], rotation=0, fontsize=16)
plt.savefig('img508b.png')


