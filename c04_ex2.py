import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

data = pd.read_table('./BAP/heatcapacity_dat.txt', header=None, sep='\s+', usecols=[1, 2])
#print(len(data))
#print(data.shape)
#print(data)

#print(np.asarray(data).T)
data = np.asarray(data).T
plt.scatter(data[0], data[1])
plt.savefig('img4ex1.png')

with pm.Model() as model_ex2:
	alpha = pm.Normal('alpha', mu=0, sd=200)
	beta = pm.Normal('beta', mu=0, sd=200)
	epsolon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, data[0])
	nu = pm.Deterministic('nu', pm.Exponential('nu_', 1/15))
	y_pred = pm.StudentT('y_pred', mu=mu, nu=nu, sd=epsolon, observed=data[1])
	
	trace_ex = pm.sample(5000)

pm.traceplot(trace_ex[500:])
plt.savefig('img4ex2.png')
pm.summary(trace_ex[500:])
pm.autocorrplot(trace_ex[500:])
plt.savefig('img4ex3.png')

pm.forestplot(trace_ex)
plt.savefig('img4ex4.png')

plt.figure()
y_temp = stats.linregress(data[0], data[1])[:2]
plt.plot(data[0], y_temp[0] * data[0] + y_temp[1], alpha=0.5)
alpha_m = trace_ex['alpha'][500:].mean()
beta_m = trace_ex['beta'][500:].mean()
plt.plot(data[0], data[0] * beta_m + alpha_m, 'g')
plt.scatter(data[0], data[1])
plt.savefig('img4ex5.png')



