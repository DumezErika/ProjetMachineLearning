import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model as lm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
sns.set_theme(style='darkgrid')

auto = pd.read_csv("Auto.csv")
tab = auto.copy()

"""Question 8"""
#reg = linear_model.LinearRegression()
horsepower = [float(i) for i in tab.loc[tab['horsepower'] != '?', 'horsepower'].tolist()]
a = np.reshape(horsepower, (-1,1))
b = tab.loc[tab['horsepower'] != '?', 'mpg'].tolist()
#reg.fit(a, b)
A = sm.add_constant(a)
lm_fit = sm.OLS(b,A).fit()
#print(lm_fit.summary())

prstd, iv_l, iv_u = wls_prediction_std(lm_fit)
fig, ax = plt.subplots(figsize=(8,6))

#ax.plot(a,b,'o', label = "data")
#ax.plot(a, lm_fit.fittedvalues, 'r--.', label="OLS")
#ax.plot(a, iv_u, 'r--')
#ax.plot(a, iv_l, 'r--')
#ax.legend(loc='best')

#sns.residplot(a, b)

#plt.show()

"""
plt.scatter(a, b, color='black')
plt.plot(a, reg.predict(a), color='blue', linewidth=2)

plt.show()
"""

#print(reg.predict([[98]]))

"""Question 9"""
#g = sns.PairGrid(tab)
#g.map(sns.scatterplot)
#plt.show()

#print(tab.corr())

temp = auto.copy()
temp = temp.loc[temp["horsepower"]!= "?"]
c = pd.to_numeric(temp["horsepower"])
temp["horsepower"] = c
X = temp.drop(['name', 'mpg'], axis = 1)
X = sm.add_constant(X)
Y = temp['mpg']
"""
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print(model.summary())
"""
res1 = smf.ols(formula='mpg ~ cylinders * displacement', data=temp).fit()
print(res1.summary())


