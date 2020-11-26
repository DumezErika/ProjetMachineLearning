import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
sns.set_theme(style="darkgrid")

auto = pd.read_csv("Auto.csv")
b = auto.copy()
b = b.loc[b["horsepower"]!= "?"]
c = pd.to_numeric(b["horsepower"])
b["horsepower"] = c
b = b.drop(["name"],axis = 1)

#g = sns.PairGrid(b)
#g.map(sns.scatterplot)
#plt.show()

matrice = b.corr()
#print(matrice)

lm_fit = smf.ols(data = b, formula = 'mpg~ cylinders + displacement + horsepower + weight +acceleration + year + origin').fit()
#print(lm_fit.summary())
#g = sns.residplot(b.cylinders, b.mpg)
#f = sns.residplot(b.displacement, b.mpg)
h1 = smf.ols(formula = 'mpg~cylinders * displacement', data= b).fit()
print(h1.summary())
#plt.show()





