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
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import SCORERS
from sklearn.metrics import neg_mean_squared_error
sns.set_theme(style='darkgrid')

college_ = pd.read_csv("College.csv", index_col=0)
college = college_.copy()
college.loc[college['Private'] == 'No', 'Private'] = int(0)
college.loc[college['Private'] == 'Yes', 'Private'] = int(1)

#a)
X = college[college.columns.difference(['Apps'])]
y = college['Apps']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=4)

#b)
ols = LinearRegression().fit(X_train,y_train)
y_pred_ols = ols.predict(X_test)
mse_ols = np.mean((y_test - y_pred_ols)**2)
print(mse_ols)

#c)
rr = RidgeCV().fit(X_train, y_train)
y_pred_rr = rr.predict(X_test)
mse_rr = np.mean((y_test - y_pred_rr)**2)
print(mse_rr)

#d)
lasso = LassoCV().fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = np.mean((y_test - y_pred_lasso)**2)
print(mse_lasso)

#e)
X_pca = PCA(n_components=2).fit(X_train).transform(X_train)
pca = PCA()
Xreg = pca.fit_transform(X_train)[:,:pc]
regr = LinearRegression()
y_cv = cross_validate(regr, Xreg, y_train)

