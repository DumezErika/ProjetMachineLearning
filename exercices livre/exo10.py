import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
sns.set_theme(style="darkgrid")

c = pd.read_csv("Carseats.csv")
# lm_fit = smf.ols(data = c, formula = 'Sales~ Price + Urban + US').fit()
# print(lm_fit.summary())
lm_fit = smf.ols(data = c, formula = 'Sales~ Price + US').fit()
#print(lm_fit.summary())


ci = lm_fit.conf_int(alpha = 0.95)
#print(ci)

model_leverage = lm_fit.get_influence().hat_matrix_diag
model_norm_residuals = lm_fit.get_influence().resid_studentized_internal
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
plt.show()