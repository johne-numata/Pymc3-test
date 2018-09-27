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

order =  5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean())/y_1.std()


plt.scatter(x_1s[0], y_1s)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.savefig('img604ex.png')


with pm.Model() as model_l:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=100)
#	beta = pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]))
	epsilon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + beta * x_1s[0]
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
	trace_l = pm.sample(2100)

chain_l = trace_l[100:]


#pm.summary(chain_l)
pm.traceplot(chain_l)
plt.savefig('img604a.png')
plt.figure()
#pm.autocorrplot(chain_l)
#plt.savefig('img604b.png')


with pm.Model() as model_p:
	alpha = pm.Normal('alpha', mu=0, sd=10)
	beta = pm.Normal('beta', mu=0, sd=1, shape=x_1s.shape[0])
	epsilon = pm.HalfCauchy('epsilon', 5)
	mu = alpha + pm.math.dot(beta, x_1s)
	y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s)
	trace_p = pm.sample(2100)

chain_p = trace_p[100:]


#pm.summary(chain_p)
pm.traceplot(chain_p)
plt.savefig('img604c.png')
plt.figure()
#pm.autocorrplot(chain_p)
#plt.savefig('img604d.png')


alpha_l_post = chain_l['alpha'].mean()
beta_l_post = chain_l['beta'].mean(axis=0)
idx = np.argsort(x_1s[0])
y_l_post = alpha_l_post + beta_l_post * x_1s[0]

plt.plot(x_1s[0][idx], y_l_post[idx], label='Linear')

alpha_p_post = chain_p['alpha'].mean()
beta_p_post = chain_p['beta'].mean(axis=0)
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], label='Pol order {}'.format(order))

plt.scatter(x_1s[0], y_1s)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend()
plt.savefig('img605.png')


print (pm.dic(trace=trace_l, model=model_l))
print( pm.dic(trace=trace_p, model=model_p))

waic_l = pm.waic(trace=trace_l, model=model_l)
waic_p = pm.waic(trace=trace_p, model=model_p)
loo_l = pm.loo(trace=trace_l, model=model_l)
loo_p = pm.loo(trace=trace_p, model=model_p)

plt.figure()
plt.subplot(121)
for idx, ic in enumerate((waic_l, waic_p)):
	plt.errorbar(ic[0], idx, xerr=ic[1], fmt='bo')
plt.title('WAIC')
plt.yticks([0, 1], ['linear', 'quadratic'])
plt.ylim(-1, 2)

plt.subplot(122)
for idx, ic in enumerate((loo_l, loo_p)):
	plt.errorbar(ic[0], idx, xerr=ic[1], fmt='go')
plt.title('LOO')
plt.yticks([0, 1], ['linear', 'quadratic'])
plt.ylim(-1, 2)
plt.tight_layout()
plt.savefig('img606.png')

