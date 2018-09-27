import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')

iris = sns.load_dataset('iris', data_home="./BAP")

df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values


with pm.Model() as model_lda:
	mus = pm.Normal('mus', mu=0, sd=10, shape=2)
	sigma = pm.HalfCauchy('sigma', 5)
	setosa = pm.Normal('setosa', mu=mus[0], sd=sigma, observed=x_0[:50])
	versicolor = pm.Normal('versicolor', mu=mus[1], sd=sigma, observed=x_0[50:])
	bd = pm.Deterministic('bd', (mus[0] + mus[1])/2)
	trace_lda = pm.sample(5000, njobs=1)

pm.traceplot(trace_lda)
plt.savefig('img514.png')

pm.summary(trace_lda)

plt.figure()
plt.axvline(trace_lda['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(trace_lda['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)
plt.plot(x_0, y_0, 'o', color='k')
plt.xlabel(x_n, fontsize=16)
plt.savefig('img515.png')


