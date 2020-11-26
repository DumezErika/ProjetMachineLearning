import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
sns.set_theme(style="darkgrid")


auto = pd.read_csv("Auto.csv")
b = auto.copy()
b = b.loc[b["horsepower"]!= "?"]
c = pd.to_numeric(b["horsepower"])
b["horsepower"] = c
#reg = lm.LinearRegression()
#x = np.reshape(b.horsepower.tolist(),(-1,1))
#y = b.mpg.tolist()
#reg.fit(x,y)
#print(reg.coef_)
#print(reg.intercept_)
#print(reg.predict([[98]]))

lm_fit = smf.ols(data = b, formula = 'mpg~horsepower').fit()
#print(lm_fit.summary())

ci = lm_fit.conf_int(alpha = 0.95)

_, iv_l, iv_u = wls_prediction_std(lm_fit)

#plt.scatter(b.horsepower, b.mpg)
#plt.plot(b.horsepower, iv_u, 'r--')
#plt.plot(b.horsepower, iv_l, 'r--')
X = pd.DataFrame({'horsepower': [b.horsepower.min(), b.horsepower.max()]})
Y_pred = lm_fit.predict(X)
#plt.plot(X, Y_pred, c='red')
g = sns.residplot(b.horsepower, b.mpg)
plt.show()