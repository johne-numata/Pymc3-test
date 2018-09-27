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

with pm.Model() as model_sf:
	alpha = pm.Normal('alpha', mu=0, sd=2, shape=2)
	beta = pm.Normal('beta', mu=0, sd=2, shape=(4, 2))
	alpha_f = tt.concatenate([[0], alpha])
	beta_f = tt.concatenate([np.zeros((4,1)), beta], axis=1)
	mu = alpha_f + pm.math.dot(x_s, beta_f)
	theta = tt.nnet.softmax(mu)
	yl = pm.Categorical('yl', p=theta, observed=y_s)
	trace_sf = pm.sample(2000, njobs=1)

pm.traceplot(trace_sf)
plt.savefig('img513.png')

pm.summary(trace_sf)

alpha_f = np.concatenate([[0], trace_sf['alpha'].mean(0)])
beta_f = np.concatenate([np.zeros((4,1)), trace_sf['beta'].mean(0)], axis=1)
data_pred = alpha_f + np.dot(x_s, beta_f)
y_pred = []
for point in data_pred:
	y_pred.append(np.exp(point)/np.sum(np.exp(point), axis=0))
print(np.sum(y_s == np.argmax(y_pred, axis=1))/len(y_s))

