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
sns.set_theme(style='darkgrid')

default_ = pd.read_csv("Default.csv")
default = default_.copy()

X = default[['income', 'balance']]
y = default[['default']]
clf = LogisticRegression().fit(X,y)


predProbs = clf.predict_proba(X)
X_design = np.hstack([np.ones((X.shape[0],1)), X])
V = np.diagflat(np.product(predProbs, axis=1))
covLogit = np.linalg.inv(np.dot(np.dot(X_design.T,V), X_design))

standard_errors = np.sqrt(np.diag(covLogit))
print("Coefficients : " , clf.coef_ , clf.intercept_)
print("Standard errors : ", standard_errors)

default.loc[default['default'] == 'No', 'default'] = int(1)
default.loc[default['default'] == 'Yes', 'default'] = int(0)

f = 'default ~ income + balance'
#model = smf.logit(str(f), default).fit()
model = smf.glm(formula = f, data = default, family = sm.families.Binomial()).fit()
print(model.summary())
