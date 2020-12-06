import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.linear_model as lm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
sns.set_theme(style='darkgrid')

weekly = pd.read_csv("Weekly.csv")

#a) 
#print(weekly.describe())
#g = sns.PairGrid(weekly)
#g.map(sns.scatterplot)
#plt.show()

#print(weekly.corr())

#b)
X = weekly.iloc[:,1:7]
y = weekly.iloc[:,8:9]
clf = LogisticRegression().fit(X,y)

#c)
y_pred = clf.predict(X)
confusion = confusion_matrix(y, y_pred)

#d)
weekly_copy = weekly.copy()
weekly_copy = weekly_copy.loc[weekly_copy["Year"] <= 2008]
A = weekly_copy[["Lag2"]]
b = weekly_copy[["Direction"]]
clf2 = LogisticRegression().fit(A,b)

weekly_copy2 = weekly.copy()
weekly_copy2 = weekly_copy2.loc[weekly_copy2["Year"] > 2008]
C = weekly_copy2[["Lag2"]]
d = weekly_copy2[["Direction"]]
d_pred = clf2.predict(C)
confusion = confusion_matrix(d, d_pred)

#e)
lda = LinearDiscriminantAnalysis()
lda.fit(A,b)
d_pred_lda = lda.predict(C)
confusion = confusion_matrix(d, d_pred_lda)

#f)
qda = QuadraticDiscriminantAnalysis()
qda.fit(A,b)
d_pred_qda = qda.predict(C)
confusion = confusion_matrix(d, d_pred_qda)

#g)
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(A,b)
d_pred_neigh = neigh.predict(C)
confusion = confusion_matrix(d, d_pred_neigh)

#i)
f = sns.boxplot(data=weekly, y='Lag4', x='Direction')
plt.show()
