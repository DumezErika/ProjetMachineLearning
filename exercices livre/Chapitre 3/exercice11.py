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

fig, ax = plt.subplots()
sns.regplot(x='x', y='y', data=df, scatter_kws={"s": 50, "alpha": 1}, ax=ax)
ax.axhline(color='gray')
ax.axvline(color='gray')

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
