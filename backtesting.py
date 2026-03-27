import numpy as np
import pandas as pd
from data_loader import load_data

# importing data
prices, returns = load_data()

# calculating portfolio returns
n_assets = returns.shape[1]
weights = np.ones(n_assets)/n_assets
portfolio_returns = returns @ weights

# historical VaR
alpha = 0.99
VaR_hist = -np.percentile(portfolio_returns, (1-alpha) * 100)
n_breaches = (portfolio_returns < -VaR_hist).sum()


from scipy.stats import chi2
# Kupiec test - formally tests whether breach rate is statistically consistent with 99% confidence level
T = len(portfolio_returns)
N = n_breaches
p = 1 - alpha

# likelihood ratio statistic follows chi-squared distribution with 1 degree of freedom
LR = -2 * (np.log((1-p)**(T-N) * p**N) - np.log((1-N/T)**(T-N) * (N/T)**N))
p_value = 1 - chi2.cdf(LR, df=1)

print(f"Kupiec LR statistic: {LR:.4f}")
print(f"P-value: {p_value:.4f}")

from statsmodels.stats.diagnostic import acorr_ljungbox

# test for independence of breaches using Ljung-Box test
breaches_binary = (portfolio_returns < -VaR_hist).astype(int)
lb_test = acorr_ljungbox(breaches_binary, lags=5)
print(lb_test)

# Basel traffic light test - counts breaches in last 250 trading days to classify model as green, amber or red
basel_window = portfolio_returns.iloc[-250:]
basel_breaches = (basel_window < -VaR_hist).sum()
print(f"Basel breaches (last 250 days): {basel_breaches}")


# Zoomed in view of 2020 crisis period to highlight breach clustering
import matplotlib.pyplot as plt
crisis_2020 = portfolio_returns.loc['2020-01-01':'2020-12-31']
breaches_2020 = crisis_2020[crisis_2020 < -VaR_hist]

plt.figure(figsize=(12, 6))
plt.plot(crisis_2020.index, crisis_2020, color='steelblue', linewidth=0.8, label='Portfolio Returns')
plt.axhline(-VaR_hist, color='red', linestyle='--', label=f'Historical VaR 99% ({-VaR_hist:.4f})')
plt.scatter(breaches_2020.index, breaches_2020, color='red', s=50, zorder=3, label=f'VaR Breaches ({len(breaches_2020)})')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('VaR Breach Clustering During COVID-19 Crisis (2020)')
plt.legend(loc='upper right')
plt.savefig('figures/breach_clustering_2020.png')
plt.show()