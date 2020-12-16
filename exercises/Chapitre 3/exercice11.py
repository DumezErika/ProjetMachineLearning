import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
sns.set_theme(style="darkgrid")

np.random.seed(1)
x = np.random.normal(size =100)
y =2*x + np.random.normal(size=100)

df = pd.DataFrame({'x': x, 'y': y})

sns.regplot(x='x', y='y')

#a)
reg = smf.ols('y ~ x + 0', df).fit()
print(reg.summary())

#b)
reg = smf.ols('x ~ y + 0', df).fit()
print(reg.summary())

#f)
reg = smf.ols('y ~ x', df).fit()
print(reg.summary())

reg = smf.ols('x ~ y', df).fit()
print(reg.summary())

plt.show()
