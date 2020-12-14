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
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import export_text
from sklearn import preprocessing
sns.set_theme(style='darkgrid')

us = pd.read_csv("USArrests.csv")
X = us

clust = AgglomerativeClustering(linkage='complete', n_clusters = 3).fit(X)
#print(clust.labels_)

X_st = preprocessing.scale(X)
clust1 = AgglomerativeClustering(linkage='complete', n_clusters = 3).fit(X_st)
print(clust1.labels_)
