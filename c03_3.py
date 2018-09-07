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

with pm.Model() as model_t:
	mu = pm.Uniform('mu', 40, 75)
	sigma = pm.HalfNormal('sigma', sd=10)
	nu = pm.Exponential('nu', 1/30)
	y = pm.StudentT('y', mu=mu, sd=sigma, nu=nu, observed=data)
	trace_t = pm.sample(1100)

chain_t = trace_t[100:]
pm.traceplot(chain_t)
plt.savefig('img308.png')

df_t = pm.summary(chain_t)

plt.clf()
y_pred = pm.sample_ppc(chain_t, 100, model_t, size=len(data))
sns.kdeplot(data, color='b')
for i in y_pred['y']:
	sns.kdeplot(i, color='r', alpha=0.1)
plt.xlim(35, 75)
plt.title("Student's t model", fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.savefig('img309.png')





