import numpy as np
import pymc3 as pm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')

iris = sns.load_dataset('iris', data_home="./BAP")

corr = iris[iris['species'] != 'virginica'].corr()
mask = np.tri(*corr.shape).T
sns.heatmap(corr, mask=mask, annot=True, cmap='viridis')
plt.savefig('img509.png')


