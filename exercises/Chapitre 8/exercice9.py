import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

oj_ = pd.read_csv("OJ.csv")
oj = oj_.copy()
oj.loc[oj['Store7'] == 'No', 'Store7'] = 0
oj.loc[oj['Store7'] == 'Yes', 'Store7'] = 1
X = oj[oj.columns.difference(['Purchase'])]
y = oj['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 800, random_state=4)
clf = DecisionTreeClassifier(random_state = 4).fit(X_train,y_train)
print(clf.get_n_leaves())
print(clf.score(X_test, y_test))
r = export_text(clf, feature_names = list(X))
#print(r)
y_pred = clf.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
