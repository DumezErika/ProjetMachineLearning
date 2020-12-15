import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

boston = pd.read_csv("Boston.csv")
X = boston[['dis']]
y = boston['nox']
poly = PolynomialFeatures(degree = 3)
X_cub = poly.fit_transform(X)
lr = LinearRegression(fit_intercept = False)
lr.fit(X_cub, y)
coefs = lr.coef_
plt.plot(X, y, '.', color = 'black')

t = np.arange(0.0, 13, 0.01)
s = coefs[0] + coefs[1]*t + coefs[2]*(t**2) + coefs[3]*(t**3)
plt.plot(t,s, color = 'red')

plt.show()
