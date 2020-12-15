import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
sns.set_theme(style='darkgrid')

auto = pd.read_csv("Auto.csv")
tab = auto.copy()

#a)
g = sns.PairGrid(tab)
g.map(sns.scatterplot)
plt.show()

#b)
print(tab.corr())

#c)
temp = auto.copy()
temp = temp.loc[temp["horsepower"]!= "?"]
c = pd.to_numeric(temp["horsepower"])
temp["horsepower"] = c
X = temp.drop(['name', 'mpg'], axis = 1)
X = sm.add_constant(X)
Y = temp['mpg']

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print(model.summary())

#e)
res1 = smf.ols(formula='mpg ~ cylinders * displacement', data=temp).fit()
print(res1.summary())
