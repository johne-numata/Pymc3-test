import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')

clusters = 3
n_cluster = [90, 50, 75]
n_total = sum(n_cluster)
means = [9, 21, 35]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

"""
sns.kdeplot(np.array(mix))
plt.xlabel('$x$', fontsize=14)
plt.savefig('img701.png')
"""

with pm.Model() as model_mg:
	p = pm.Dirichlet('p', a=np.ones(clusters))
	
	means = pm.Normal('means', mu=[10, 25, 30], sd=2, shape=clusters)
	sd = pm.HalfCauchy('sd', 5)
	y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
	trace_mg = pm.sample(10000, njobs=1)

chain_mg = trace_mg[1000:]
varname_mg = ['means', 'sd', 'p']
pm.traceplot(chain_mg, varname_mg)
#pm.traceplot(chain_mg)
plt.savefig('img705_b.png')

#pm.summary(chain_mg, varname_mg)

plt.figure()
ppc = pm.sample_ppc(chain_mg, 100, model_mg)
#print(ppc)
#for i in ppc['y']:
#	sns.kdeplot(i, alpha=0.1, color='b')
sns.kdeplot(ppc['y'], alpha=0.1, color='b')

sns.kdeplot(np.array(mix), lw=2, color='k')
plt.xlabel('$x$', fontsize=14)
plt.savefig('img706_b.png')
