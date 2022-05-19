import statsmodels as sm
import pandas as pd

import statsmodels.stats.api as sms
import statsmodels.api as sm

from scipy import stats
from statsmodels.api import OLS

def check_OLS_hypothesis(X, Y):
    """
    Functions that check the hypothesis of an OLS model: normality, homodescaticity, stationarity
    param X: independant variable
    param Y: dependant variable
    return: 
    """

    # get residuals
    model = OLS(X, Y)
    results = model.fit()
    residuals = Y - results.params[0]

    ### Step 1: Calculate pvalue for each metrics
    metrics = [] 
    metrics.append(('Jarque-Bera', 'Normality', stats.jarque_bera(residuals)[1]))
    metrics.append(('Shapiro-Wilk', 'Normality', stats.shapiro(residuals)[1]))
    metrics.append(('Kolmogorov-Smirnov', 'Normality',
        stats.kstest(residuals, 'norm')[1]))
    metrics.append(('Goldfield-Quandt', 'Homogeneity', sms.het_goldfeldquandt(results.resid, results.model.exog)[1]))
    metrics.append(('Breusch-Pagan', 'Homogeneity', sms.het_breuschpagan(results.resid, results.model.exog)[1]))
    #  metrics.append(('White', 'Homogeneity', sms.het_white(results.resid, results.model.exog)[1]))
    metrics.append(('Durbin-Watson', 'Stationarity', sms.durbin_watson(results.resid)))
    metrics.append(('Breusch-Godfrey', 'Stationarity', sms.acorr_breusch_godfrey(results, nlags=1)[3]))
          
    ### Step 2: Print pvalue results 
    for name, test, pvalue in metrics:
        print(f"[Test for {test}] p-value of {name} is: {pvalue}")

    ### Step 3: Hypothesis testing results




