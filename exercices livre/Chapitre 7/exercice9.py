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
sns.set_theme(style='darkgrid')

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
