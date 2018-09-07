import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(123)
n_experiments = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)

print(data)

with pm.Model() as first_model:
	theta = pm.Beta('theta', alpha=1, beta=1)
#	theta = pm.Uniform('theta', lower=0, upper=1)
	y = pm.Bernoulli('y', p=theta, observed=data)
	start = pm.find_MAP()
	step = pm.Metropolis()
	trace = pm.sample(1000, step=step)

burnin = 100
chaine = trace[burnin:]
pm.traceplot(chaine, lines={'theta':theta_real});
plt.savefig('img204.png')


with first_model:
	multi_trace = pm.sample(1000, step=step, njobs=4)

burnin = 100
multi_chaine = multi_trace[burnin:]
pm.traceplot(multi_chaine, lines={'theta':theta_real});
plt.savefig('img206.png')

pm.gelman_rubin(multi_chaine)
{'theta': 1.0074579751170656, 'theta_logodds': 1.009770031607315}
pm.forestplot(multi_chaine, varnames={'theta'});
plt.savefig('img207.png')

print (pm.summary(multi_chaine))

pm.autocorrplot(multi_chaine)
plt.savefig('img208.png')

print(pm.effective_n(multi_chaine)['theta'])

pm.plot_posterior(chaine, kde_plot=True, rope=[0.45, 0.55], ref_val=0.5)
plt.savefig('img209.png')

