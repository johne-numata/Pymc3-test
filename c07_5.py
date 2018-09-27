import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')

np.random.seed(42)
n = 100
theta = 2.5
pi = 0.1

count = np.array([(np.random.random() > pi) * np.random.poisson(theta) for i in range(n)])

with pm.Model() as ZIP:
	psi = pm.Beta('psi', 1, 1)
	lam = pm.Gamma('lam', 2, 0.1)
	
	y = pm.ZeroInflatedPoisson('y', psi, lam, observed=count)
	trace_ZIP = pm.sample(5000, njobs=1)

chain_ZIP = trace_ZIP[100:]
pm.traceplot(chain_ZIP)
plt.savefig('img706.png')


