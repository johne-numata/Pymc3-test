import numpy as np
#import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#for i in [1,2,3,4,5]: print(np.mean(stats.t(loc=0, scale=1, df=1).rvs(100)))
#for i in [1,2,3,4,5]: print(np.mean(stats.t(loc=0, scale=1, df=100).rvs(100)))

x_values = np.linspace(-10, 10, 200)
for df in [1, 2, 5, 30]:
	distri = stats.t(df)
	x_pdf = distri.pdf(x_values)
	plt.plot(x_values, x_pdf, label=r'$\nu$ = {}'.format(df))
x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, label=r'$\nu = \infty$')
plt.xlabel('$x$')
plt.ylabel('$p(x)&')
plt.legend(loc=0, fontsize=14)
plt.xlim(-7, 7)
plt.savefig('img306.png')

