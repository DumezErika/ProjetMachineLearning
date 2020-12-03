import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

""" a) """
college = pd.read_csv("College.csv", index_col=0)

""" c) """
print(college.describe())

#ii
a = college.iloc[:,0:10]
g = sns.PairGrid(a)
g.map(sns.scatterplot)

#iii
f = sns.boxplot(data=college, x= 'Private', y= 'Outstate')

#iv
b = college.copy()
b['Elite'] = 'No'
b.loc[b['Top10perc']>50,'Elite'] = 'Yes'
print(b.describe(include = ['object']))
h = sns.boxplot(data = b, x = 'Elite', y= 'Outstate')

#v
f = sns.histplot(data = b, x = 'Top10perc')

plt.show()