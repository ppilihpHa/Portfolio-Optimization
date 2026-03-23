import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#   HELPFUNCTIONS

#   easier to quickly check different vars
def info(info):  
    for elem in info:
        print('------------')
        print(elem)
        print('------------')

#   easiert when done multiple ports
def get_portfolio_metrics(weights, mean_return, cov_mat):
    ret = np.dot(weights, mean_return) # used with optimal weights
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    return ret, volatility

#   Data

stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
year = int(input('year:'))
data = yf.download(stocks, start=f'{year}-01-01', end=f'{year + 1}-01-01')
data = data['Close'].dropna()

#   RETURNS

log_returns = np.log(data / data.shift(1)).dropna()
mean_log_return = log_returns.mean()
cov_mat = log_returns.cov()

mean_log_return_an = mean_log_return * 252 # annualisiert, ca. 252 Handelstage laut Internet
cov_mat_an = cov_mat * 252

#   CONSTRAINTS (for minimization Problem)

n = len(stocks)
bounds = tuple((0,1) for i in range(n)) # [0,1] boundary for each stock
constraints = {'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1} 

#   OPTIMIZATION

inital_guess = n * [1./n,]

def pf_variance(weights, cov_mat):
        return np.dot(weights.T, np.dot(cov_mat, weights))

results = minimize(pf_variance, inital_guess, args=(cov_mat_an,), method='SLSQP', bounds=bounds, constraints=constraints)

opt_weights = results.x
opt_return, opt_volatility = get_portfolio_metrics(opt_weights, mean_log_return_an, cov_mat_an)

#info([data.head(5), log_returns, mean_log_return, cov_mat])
#info([cov_mat_an])
info([stocks ,opt_weights.round(3), opt_return, opt_volatility])
