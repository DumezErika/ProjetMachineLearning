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

def boot_fn(dataFrame, index):
    df = dataFrame.iloc[index, :]
    X = df[['income', 'balance']]
    y = df['default']
    clf = LogisticRegression().fit(X,y)
    return(clf.coef_, clf.intercept_)
    
def boot_se(dataFrame, index):
    df = dataFrame.iloc[index,:]
    X = df[['income', 'balance']]
    y = df['default']
    clf = LogisticRegression().fit(X,y)
    predProbs = clf.predict_proba(X)
    X_design = np.hstack([np.ones((X.shape[0],1)), X])
    V = np.diagflat(np.product(predProbs, axis=1))
    covLogit = np.linalg.inv(np.dot(np.dot(X_design.T,V), X_design))
    return(np.sqrt(np.diag(covLogit)))

def boot(dataFrame, n):
    standard_errors = []
    for i in range(n):
        index = np.random.choice([i for i in range(len(dataFrame))], len(dataFrame))
        standard_errors.append(boot_se(dataFrame, index))
    return pd.DataFrame(standard_errors).mean()

print(boot(default,100))
