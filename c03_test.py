import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

N_samples = [30, 30, 30]
G_samples = [18, 18, 18]

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
	data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i] - G_samples[i]]))
print(group_idx)
print(data)

with pm.Model() as model_h:
	alpha = pm.HalfCauchy('alpha', beta = 10)
	beta = pm.HalfCauchy('beta', beta = 10)
	theta = pm.Beta('theta', alpha, beta, shape=len(N_samples))
	y = pm.Bernoulli('y', p=theta[group_idx], observed=data)

	t = pm.Beta('t', 10, 10, shape=len(N_samples))
	print(t.random(size=10))


#	trace_h = pm.sample(2000)
#chain_h = trace_h[200:]
#pm.traceplot(chain_h)
#plt.savefig('img314.png')

#t = pm.Beta.dist(alpha, beta, shape=len(N_samples))
#print(theta.dist.random(size=10))


