import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

default_ = pd.read_csv("Default.csv")
default = default_.copy()

# a
# Première approche
X = default[['income', 'balance']]
y = default['default']
clf = LogisticRegression().fit(X,y)

predProbs = clf.predict_proba(X)
X_design = np.hstack([np.ones((X.shape[0],1)), X])
V = np.diagflat(np.product(predProbs, axis=1))
covLogit = np.linalg.inv(np.dot(np.dot(X_design.T,V), X_design))

standard_errors = np.sqrt(np.diag(covLogit))
print("Coefficients : " , clf.coef_ , clf.intercept_)
print("Standard errors : ", standard_errors)

#Deuxième approche
default.loc[default['default'] == 'No', 'default'] = int(1)
default.loc[default['default'] == 'Yes', 'default'] = int(0)

f = 'default ~ income + balance'
model = smf.glm(formula = f, data = default, family = sm.families.Binomial()).fit()
print(model.summary())

# b
def boot_fn(dataFrame, index):
    df = dataFrame.iloc[index, :]
    X = df[['income', 'balance']]
    y = df['default']
    clf = LogisticRegression().fit(X,y)
    return(clf.coef_, clf.intercept_)

# c
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

