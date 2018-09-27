import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

iris = sns.load_dataset('iris', data_home="./BAP")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
y_0 = np.concatenate((y_0, np.ones(6)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.3]))
x_0_m = x_0 - x_0.mean()
plt.plot(x_0, y_0, 'o', color= 'k')
plt.savefig('img712.png')

with pm.Model() as model_reg:
	alpha_tmp = pm.Normal('alpha_tmp', mu=0, sd=100)
#	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10)
	mu = alpha_tmp + beta * x_0_m
	theta = pm.Deterministic('theta', 1/(1 + pm.math.exp(-mu)))
	
	pi = pm.Beta('pi', 1, 1)
	p  = pi * 0.5 + (1 - pi) * theta
	
	alpha = pm.Deterministic('alpha', alpha_tmp - beta * x_0.mean())
	bd = pm.Deterministic('bd', -alpha/beta)
	
	yl = pm.Bernoulli('yl', p=p, observed=y_0)
	trace_rlg = pm.sample(5000)

chain_rlg = trace_rlg[1000:]
#varnames = ['alpha', 'beta', 'bd']
pm.traceplot(chain_rlg)
#plt.savefig('img505c.png')
plt.savefig('img713.png')

#pm.summary(trace_rlg)

plt.figure()
plt.plot(x_0, y_0, 'o', color='k')

theta = trace_rlg['theta'].mean(axis=0)
idx = np.argsort(x_0)
plt.plot(x_0[idx], theta[idx], color='b', lw=3)
plt.axvline(trace_rlg['bd'].mean(), ymax=1, color='r')
bd_hpd = pm.hpd(trace_rlg['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='r', alpha=0.5)

theta_hpd = pm.hpd(trace_rlg['theta'])[idx]
plt.fill_between(x_0[idx], theta_hpd[:, 0], theta_hpd[:, 1], color='b', alpha=0.5)
plt.xlabel(x_n, fontsize=16)
plt.ylabel('$\\theta$', rotation=90, fontsize=16)
plt.savefig('img714.png')

