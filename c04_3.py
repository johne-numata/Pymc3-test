import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

np.random.seed(314)
N = 100
alfa_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

with pm.Model() as model:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=10)
	epsilon = pm.HalfCauchy('epsilon', 5)
	
	mu = alpha + beta * x
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)
	
	rb = pm.Deterministic('rb', (beta * x.std() / y.std()) ** 2)
		
	y_mean = y.mean()
	ss_reg = pm.math.sum((mu - y_mean)**2)
	ss_tot = pm.math.sum((y - y_mean)**2)
	rss = pm.Deterministic('rss', ss_reg / ss_tot)	
	
#	start = pm.find_MAP()
#	step = pm.NUTS()
#	trace_n = pm.sample(2000, step, start)
	trace_n = pm.sample(2000)

pm.traceplot(trace_n)
plt.savefig('img411.png')

varnames = ['alpha', 'beta', 'epsilon', 'rb', 'rss']
pm.summary(trace_n, varnames)

