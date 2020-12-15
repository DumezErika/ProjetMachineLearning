import pandas as pd
from random import sample
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

us = pd.read_csv("USArrests.csv")
X = us

clust = AgglomerativeClustering(linkage='complete', n_clusters = 3).fit(X)
#print(clust.labels_)

X_st = preprocessing.scale(X)
clust1 = AgglomerativeClustering(linkage='complete', n_clusters = 3).fit(X_st)
print(clust1.labels_)
