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
eps_real = np.random.normal(0, 0.5, size= N)

x = np.random.normal(10, 1, N)
y_real = alfa_real + beta_real * x
y = y_real + eps_real

data = np.stack((x, y)).T
#print(data)

with pm.Model() as pearson_model:
	mu = pm.Normal('mu', mu=data.mean(0), sd = 10, shape = 2)
	sigma_1 = pm.HalfNormal('sigma_1', 10)
	sigma_2 = pm.HalfNormal('sigma_2', 10)
	rho = pm.Uniform('rho', -1, 1)
	cov = pm.math.stack(([sigma_1**2, sigma_1 * sigma_2 * rho], [sigma_1 * sigma_2 * rho, sigma_2**2]))
	y_pred = pm.MvNormal('y_pres', mu=mu, cov = cov, observed=data)
	
	start = pm.find_MAP()
	step = pm.NUTS(scaling= start)
	trace_p = pm.sample(1000, step = step, stsrt = start, njobs=1)

pm.traceplot(trace_p)
pm.summary(trace_p)
plt.savefig('img413.png')
