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

with pm.Model() as model_ug:
	p = pm.Dirichlet('p', a=np.ones(clusters))
	category = pm.Categorical('category', p=p, shape=n_total)
	
	means = pm.Normal('means', mu=[10, 25, 30], sd=2, shape=clusters)
	sd = pm.HalfCauchy('sd', 5)
	y = pm.Normal('y', mu=means[category], sd=sd, observed=mix)
	trace_ug = pm.sample(10000, njobs=1)

chain_ug = trace_ug[1000:]
varname_ug = ['means', 'sd', 'p']
pm.traceplot(chain_ug, varname_ug)
plt.savefig('img705.png')

# pm.summary(chain_ug, varname_ug)

plt.figure()
ppc = pm.sample_ppc(chain_ug, 50, model_ug)
for i in ppc['y']:
	sns.kdeplot(i, alpha=0.1, color='b')

sns.kdeplot(np.array(mix), lw=2, color='k')
plt.xlabel('$x$', fontsize=14)
plt.savefig('img706.png')
