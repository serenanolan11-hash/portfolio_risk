import numpy as np
import pandas as pd
from data_loader import load_data

# load prices and log returns
prices, returns = load_data()
cov_matrix = returns.cov().values

# create a min variance portfolio
from scipy.optimize import minimize
def objective(weights):
    return weights @ cov_matrix @ weights

n_assets = returns.shape[1]
guess = np.ones(n_assets) / n_assets

bounds = tuple((0,1) for x in range(n_assets))

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

result = minimize(objective, guess, method='SLSQP', bounds=bounds, constraints=constraints)
print(result)


# filtering dates to a crisis period
crisis_returns = returns.loc['2020-01-01' : '2020-12-31']

# calculating portfolio returns for crisis period using min variance weights
portfolio_crisis_returns = crisis_returns @ result.x

# calcualting VaR and ES during crisis period
alpha = 0.99
VaR_crisis = -np.percentile(portfolio_crisis_returns, (1-alpha) * 100)
VaR_crisis_threshold = np.percentile(portfolio_crisis_returns, (1-alpha) * 100)
ES_crisis = -portfolio_crisis_returns[portfolio_crisis_returns <= VaR_crisis_threshold].mean()

# comparison graph for regular vs crisis period
portfolio_returns = returns @ result.x
VaR_hist = -np.percentile(portfolio_returns, (1 - alpha) * 100)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(crisis_returns.index, portfolio_crisis_returns, color='steelblue', linewidth=0.8, label='Crisis Portfolio Returns')
plt.axhline(-VaR_hist, color='red', linestyle='--', label=f'Normal VaR 99% ({-VaR_hist:.4f})')
plt.axhline(-VaR_crisis, color='purple', linestyle='--', label=f'Crisis VaR 99% ({-VaR_crisis:.4f})')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Portfolio Returns During COVID-19 Crisis (2020) with VaR Thresholds')
plt.legend(loc='upper right')
plt.savefig('figures/crisis_chart.png')
plt.show()


# comparison table
ES_hist = -portfolio_returns[portfolio_returns <= -VaR_hist].mean()

comparison = pd.DataFrame({
    'VaR (99%)': [VaR_hist, VaR_crisis],
    'ES (99%)': [ES_hist, ES_crisis]
}, index=['Normal Period', 'Crisis Period (2020)'])

print(comparison)