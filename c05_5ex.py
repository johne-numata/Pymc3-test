import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')

iris = sns.load_dataset('iris', data_home="./BAP")

y_s = pd.Categorical(iris['species']).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values
x_s = (x_s - x_s.mean(axis=0))/x_s.std(axis=0)

with pm.Model() as model_s:
	alpha = pm.Normal('alpha', mu=0, sd=2)
	beta = pm.Normal('beta', mu=0, sd=2, shape=4)
	mu = alpha + pm.math.dot(x_s, beta)
	epsilon = pm.HalfCauchy('epsilon', 5)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_s)
	trace_s = pm.sample(2000, njobs=1)

pm.traceplot(trace_s)
plt.savefig('img517ex.png')

pm.summary(trace_s)

data_pred = trace_s['alpha'].mean(0) + np.dot(x_s, trace_s['beta'].mean(0))
print(np.sum(y_s == np.round(data_pred))/len(y_s))

