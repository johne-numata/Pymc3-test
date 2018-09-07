import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

N_samples = [30, 30, 30]
G_samples = [18, 3, 3]

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
	data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i] - G_samples[i]]))
#print(group_idx)
#print(data)

with pm.Model() as model_h:
	alpha = pm.HalfCauchy('alpha', beta = 10)
	beta = pm.HalfCauchy('beta', beta = 10)
	theta = pm.Beta('theta', alpha, beta, shape=len(N_samples))
	y = pm.Bernoulli('y', p=theta[group_idx], observed=data)
	trace_h = pm.sample(2000)
	
chain_h = trace_h[200:]
pm.traceplot(chain_h)
pm.summary(chain_h)
plt.savefig('img314.png')
plt.clf()
print(chain_h)


x = np.linspace(0, 1, 100)
for i in np.random.randint(0, len(chain_h), size=100):
	pdf = stats.beta(chain_h['alpha'][i], chain_h['beta'][i]).pdf(x)
	plt.plot(x, pdf, 'g', alpha=0.5)

dist = stats.beta(chain_h['alpha'].mean(), chain_h['beta'].mean())
pdf = dist.pdf(x)
mode = x[np.argmax(pdf)]
mean = dist.moment(1)
plt.plot(x, pdf, label='mode={:.2f}\nmean={:2f}'.format(mode, mean))

plt.legend(fontsize=14)
plt.xlabel(r'$\theta_{prior}$', fontsize=16)
plt.savefig('img315.Png')
