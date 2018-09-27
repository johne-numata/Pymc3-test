import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')

real_alpha = 4.25
real_beta = [8.7, -1.2]
data_size = 20
nois = np.random.normal(0, 2, size=data_size)
x_1 = np.linspace(0, 5, data_size)
y_1 = real_alpha + real_beta[0] * x_1 + real_beta[1] * x_1 **2 + nois

order = 2 # 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean())/y_1.std()

with pm.Model() as model_l:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=1)
	epsilon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + beta * x_1s[0]
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
	trace_l = pm.sample(2100)

chain_l = trace_l[100:]


with pm.Model() as model_p:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=1, shape=x_1s.shape[0])
	epsilon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, x_1s)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
	trace_p = pm.sample(2100)

chain_p = trace_p[100:]

alpha_l_post = chain_l['alpha'].mean()
beta_l_post = chain_l['beta'].mean(axis=0)
idx = np.argsort(x_1s[0])
y_l_post = alpha_l_post + beta_l_post * x_1s[0]

alpha_p_post = chain_p['alpha'].mean()
beta_p_post = chain_p['beta'].mean(axis=0)
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)

plt.subplot(121)
plt.scatter(x_1s[0], y_1s, c='r')
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Linear')
for i in range(0, len(chain_l['alpha']), 50):
	plt.scatter(x_1s[0], chain_l['alpha'][i] + chain_l['beta'][i] * x_1s[0], 50, c='g', alpha=1), 
plt.subplot(122)
plt.scatter(x_1s[0], y_1s, c='r')
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Order {}'.format(order))

for i in range(0, len(chain_p['alpha']), 50):
	plt.scatter(x_1s[0], chain_p['alpha'][i] + np.dot(chain_p['beta'][i], x_1s), c='g', edgecolor='g', alpha=0.5)
idx = np.argsort(x_1)
plt.plot(x_1s[0][idx], alpha_p_post + np.dot(beta_p_post, x_1s)[idx], c='g', alpha=1)
plt.savefig('img607.png')


