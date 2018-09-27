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
data_size = 500
nois = np.random.normal(0, 2, size=data_size)
x_1 = np.linspace(0, 5, data_size)
y_1 = real_alpha + real_beta[0] * x_1 + real_beta[1] * x_1 **2 + nois

order = 5 # 5
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
	beta = pm.Normal('beta', mu=0, sd=1, shape=2)
	epsilon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, x_1s[0:2])
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
	trace_p = pm.sample(2100)

chain_p = trace_p[100:]

with pm.Model() as model_t:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=1, shape=x_1s.shape[0])
	epsilon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, x_1s)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
	trace_t = pm.sample(2100)

chain_t = trace_t[100:]

pm.traceplot(chain_l)
plt.savefig('img604ex_2.png')
plt.figure()
pm.traceplot(chain_p)
plt.savefig('img604ex_3.png')
plt.figure()
pm.traceplot(chain_t)
plt.savefig('img604ex_4.png')
plt.figure()

waic_l = pm.waic(trace=trace_l, model=model_l)
waic_t = pm.waic(trace=trace_t, model=model_t)
waic_p = pm.waic(trace=trace_p, model=model_p)
loo_l = pm.loo(trace=trace_l, model=model_l)
loo_t = pm.loo(trace=trace_t, model=model_t)
loo_p = pm.loo(trace=trace_p, model=model_p)

plt.figure()
plt.subplot(121)
for idx, ic in enumerate((waic_l, waic_p, waic_t)):
	plt.errorbar(ic[0], idx, xerr=ic[1], fmt='bo')
plt.title('WAIC')
plt.yticks([0, 1, 2], ['linear', 'quadratic', 'tetra'])
plt.ylim(-1, 3)

plt.subplot(122)
for idx, ic in enumerate((loo_l, loo_p, loo_t)):
	plt.errorbar(ic[0], idx, xerr=ic[1], fmt='go')
plt.title('LOO')
plt.yticks([0, 1, 2], ['linear', 'quadratic', 'tetra'])
plt.ylim(-1, 3)
plt.tight_layout()
plt.savefig('img606ex.png')



alpha_t_post = chain_t['alpha'].mean()
beta_t_post = chain_t['beta'].mean(axis=0)
y_t_post = alpha_t_post + np.dot(beta_t_post, x_1s)

plt.figure()
plt.scatter(x_1s[0], y_1s, c='r')
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Order {}'.format(3))

for i in range(0, len(chain_t['alpha']), 50):
	plt.scatter(x_1s[0], chain_t['alpha'][i] + np.dot(chain_t['beta'][i], x_1s), c='g', edgecolor='g', alpha=0.5)
idx = np.argsort(x_1)
plt.plot(x_1s[0][idx], alpha_t_post + np.dot(beta_t_post, x_1s)[idx], c='g', alpha=1)
plt.savefig('img607ex.png')

