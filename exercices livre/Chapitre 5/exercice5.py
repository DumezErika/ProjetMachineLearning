import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

default = pd.read_csv("Default.csv")

#a)
X = default[['income', 'balance']]
y = default['default']
clf = LogisticRegression().fit(X,y)

#b)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=4)
clf1 = LogisticRegression().fit(X_train, y_train)
#print(confusion_matrix(y_test, clf1.predict(X_test)))

#c)
default_copy = default.copy()
default_copy.loc[default_copy['student'] == 'No', 'student'] = 0
default_copy.loc[default_copy['student'] == 'Yes', 'student'] = 1
A = default_copy[['income', 'balance', 'student']]
b = default_copy['default']
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.33, random_state=4)
clf2 = LogisticRegression().fit(A_train,b_train)
print(confusion_matrix(b_test, clf2.predict(A_test)))
