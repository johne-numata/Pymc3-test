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
	bd = (mus[0] + mus[1])/2
	trace_lda = pm.sample(5000, njobs=1)

ppc_lda = pm.sample_ppc(trace_lda, 100, model_lda, vars=[bd]) 

with pm.Model() as model_0:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10)
	mu = alpha + pm.math.dot(x_0, beta)
	theta = pm.Deterministic('theta', 1/(1 + pm.math.exp(-mu)))
	bd = -alpha/beta
	
	yl = pm.Bernoulli('yl', p=theta, observed=y_0)
	trace_0 = pm.sample(5000)
#	ppc_0 = pm.sample_ppc(trace=trace_0, samples=100, vars=[bd]) 


#ppc_0 = pm.sample_ppc(trace_0, 100, model_0, vars=[bd])

#print(ppc_0)
plt.subplot(1, 2, 1)
#for tr in ppc_0['bd']:
#	sns.kdeplot(tr)
#plt.title("logistic")

print(ppc_lda)
plt.subplot(1, 2, 2)
for tr in ppc_lda['bd']:
	sns.kdeplot(tr)
plt.title("linear discriminant analysis")
plt.savefig('img516ex.png')
