import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')

fish_data = sns.load_dataset('fish', data_home="./BAP")

#print(fish_data)
#print(fish_data.describe())

with pm.Model() as ZIP_reg:
	psi = pm.Beta('psi', 1, 1)
	
	alpha = pm.Normal('alpha', 0, 10)
	beta = pm.Normal('beta', 0, 10, shape=2)
	lam = pm.math.exp(alpha + beta[0] * fish_data['child'] + beta[1] * fish_data['camper'])
	
	y = pm.ZeroInflatedPoisson('y', psi, lam, observed=fish_data['count'])
	trace_ZIP_reg = pm.sample(2000, njobs=1)

chain_ZIP_reg = trace_ZIP_reg[100:]
pm.traceplot(chain_ZIP_reg)
plt.savefig('img710.png')

plt.figure()
pm.autocorrplot(chain_ZIP_reg)
plt.savefig('img710b.png')

plt.figure()
children = [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
thin = 5
for n in children:
#	without_camper = chain_ZIP_reg['alpha'][::thin] + chain_ZIP_reg['beta'][:, 0][::thin] * n
#	with_camper = without_camper + chain_ZIP_reg['beta'][:, 1][::thin]
	without_camper = chain_ZIP_reg['alpha'] + chain_ZIP_reg['beta'][:, 0] * n
	with_camper = without_camper + chain_ZIP_reg['beta'][:, 1]
	fish_count_pred_0.append(np.exp(without_camper))
	fish_count_pred_1.append(np.exp(with_camper))

plt.plot(children, fish_count_pred_0, 'bp', alpha=0.01)
plt.plot(children, fish_count_pred_1, 'ro', alpha=0.01)

plt.xticks(children)
plt.xlabel('Number of children', fontsize=14)
plt.ylabel('Fish caught', fontsize=14)
plt.plot([], 'bo', label='without camper')
plt.plot([], 'ro', label= 'with camper')
plt.legend(fontsize=14)
plt.savefig('img711.png')

