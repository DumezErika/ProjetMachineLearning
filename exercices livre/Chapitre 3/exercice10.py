import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

c = pd.read_csv("Carseats.csv")

#a)
lm_fit = smf.ols(data = c, formula = 'Sales~ Price + Urban + US').fit()
print(lm_fit.summary())

#e)
lm_fit = smf.ols(data = c, formula = 'Sales~ Price + US').fit()
print(lm_fit.summary())

#g)
ci = lm_fit.conf_int(alpha = 0.95)
print(ci)

#h)
model_leverage = lm_fit.get_influence().hat_matrix_diag
model_norm_residuals = lm_fit.get_influence().resid_studentized_internal
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
plt.show()
