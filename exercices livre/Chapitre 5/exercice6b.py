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
    y = df[['default']]
    clf = LogisticRegression().fit(X,y)
    return(clf.coef_, clf.intercept_)

print(boot_fn(default, [i for i in range(10000)]))
