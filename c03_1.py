import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#plt.style.use('seaborn-darkgrid')
#np.set_printoptions(precision=2)
#pd.set_option('display.precision', 2)

data = np.array([51.06, 55.12, 53.73, 50.24, 52.05, 56.40, 48.45, 52.34,
55.65, 51.49, 51.86, 63.43, 53.00, 56.09, 51.93, 52.31, 52.33, 57.48, 
57.44, 55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73, 51.94, 
54.95, 50.39, 52.91, 51.50, 52.68, 47.72, 49.73, 51.82, 54.99, 52.84, 
53.19, 54.52, 51.46, 53.73, 51.61, 49.81, 52.42, 54.30, 53.84, 53.16])

sns.kdeplot(data)
plt.savefig('img302.png')

with pm.Model() as model_g:
#	mu = pm.Normal('mu', mu=100, sd=10)
	mu = pm.Uniform('mu', 40, 75)
	sigma = pm.HalfNormal('sigma', sd=10)
	y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
	trace = pm.sample(1100)

chain_g = trace[100:]
pm.traceplot(chain_g)
plt.savefig('img303.png')

df = pm.summary(chain_g)

plt.clf()
y_pred = pm.sample_ppc(chain_g, 100, model_g, size=len(data))
#print(np.asarray(y_pred['y']).shape)
sns.kdeplot(data, color='b')
for i in y_pred['y']:
	sns.kdeplot(i, color='r', alpha=0.1)
plt.xlim(35, 75)
plt.title('Gaussian model', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.savefig('img305.png')





