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

with pm.Model() as model_kg:
	p = pm.Dirichlet('p', a=np.ones(clusters))
	category = pm.Categorical('category', p=p, shape=n_total)
	
	means = pm.math.constant([10, 20, 35])
	y = pm.Normal('y', mu=means[category], sd=2, observed=mix)
	trace_kg = pm.sample(10000, njobs=1)

chain_kg = trace_kg[1000:]
varname_kg = ['p']
pm.traceplot(chain_kg, varname_kg)
plt.savefig('img704.png')
