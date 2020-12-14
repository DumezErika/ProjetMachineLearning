import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from random import sample
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model as lm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.metrics import SCORERS
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
sns.set_theme(style='darkgrid')

boston = pd.read_csv("Boston.csv")
X = boston[['dis']]
y = boston['nox']
mse_tab = []

for i in range(2,11):
    poly = PolynomialFeatures(degree = i)
    X_deg = poly.fit_transform(X)
    lr = LinearRegression(fit_intercept = False)
    lr.fit(X_deg, y)
    coefs = lr.coef_
    plt.figure(i)
    plt.plot(X, y, '.', color = 'black')
    t = np.arange(0.0, 13, 0.01)
    s = coefs[0]
    j = 1
    while j <= i:
        s += coefs[j]*(t**j)
        j += 1
    plt.plot(t,s, color = 'red')
    #plt.show()
    y_pred = lr.predict(X_deg)
    mse_tab.append(len(y)*mean_squared_error(y, y_pred))

#c)

def f(i):
    poly = PolynomialFeatures(degree = i)
    X_deg = poly.fit_transform(X)
    lr = LinearRegression(fit_intercept = False)
    lr.fit(X_deg, y)
    y_pred = cross_val_predict(lr, X_deg, y)
    return(mean_squared_error(y, y_pred))
    
deg = 2
min = f(deg)
for i in range(2,11):
    if f(i) < min:
        min = f(i)
        deg = i
print(deg)
