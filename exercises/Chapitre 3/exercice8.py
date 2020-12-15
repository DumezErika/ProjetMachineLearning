import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
sns.set_theme(style="darkgrid")


auto = pd.read_csv("Auto.csv")
df = auto.copy()
df = df.loc[b["horsepower"]!= "?"]
temp = pd.to_numeric(df["horsepower"])
df["horsepower"] = temp

#a
#Première approche
reg = lm.LinearRegression()
X = np.reshape(df.horsepower.tolist(),(-1,1))
y = df.mpg.tolist()
reg.fit(X,y)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[98]]))

#Deuxième approche
lm_fit = smf.ols(data = df, formula = 'mpg~horsepower').fit()
print(lm_fit.summary())
_, iv_l, iv_u = wls_prediction_std(lm_fit)

#b
plt.scatter(df.horsepower, df.mpg)
plt.plot(df.horsepower, iv_u, 'r--')
plt.plot(df.horsepower, iv_l, 'r--')
X = pd.DataFrame({'horsepower': [df.horsepower.min(), df.horsepower.max()]})
y_pred = lm_fit.predict(X)
plt.plot(X, y_pred, c='red')

#c
g = sns.residplot(df.horsepower, df.mpg)
plt.show()