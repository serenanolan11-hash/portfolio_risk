import numpy as np
import pandas as pd
from data_loader import load_data

# load prices and log returns
prices, returns = load_data()

# producing an equally weighted portfolio
n_assets = returns.shape[1]
weights = np.ones(n_assets) / n_assets

portfolio_returns = returns @ weights

# historical VaR
alpha = 0.99
VaR_hist = -np.percentile(portfolio_returns, (1 - alpha) * 100)

# historical ES
VaR_threshold = np.percentile(portfolio_returns, (1 - alpha) * 100)
ES = -portfolio_returns[portfolio_returns <= VaR_threshold].mean()

print("Historical VaR (99%):", -VaR_threshold)
print("Historical ES (99%):", ES)


# Monte Carlo VaR and ES
n_sims = 10000
mu = returns.mean().values
cov_matrix = returns.cov().values

print(np.isnan(mu).any(), np.isnan(cov_matrix).any())
# simulate 10,000 draws from a multivariate normal distribution with chosen mu and cov_matrix
simulated_returns = np.random.multivariate_normal(mu, cov_matrix, n_sims)

# simulated portfolio returns for equally weighted portfolio
sim_port_returns = simulated_returns @ weights

VaR_MC = -np.percentile(sim_port_returns, (1 - alpha) * 100)
VaR_threshold_MC = np.percentile(sim_port_returns, (1 - alpha) * 100)
ES_MC = -sim_port_returns[sim_port_returns <= VaR_threshold_MC].mean()

print("MC VaR (99%):", -VaR_threshold_MC)
print("MC ES (99%):", ES_MC)

 
# histogram to show monte comparison of Monte Carlo and Historical VaR and ES estimates
import matplotlib.pyplot as plt 
plt.figure(figsize=(12, 6))
plt.hist(sim_port_returns, bins=100) 
plt.xlabel("Simulated Portfolio Return")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulated Portfolio Return Distribution")
plt.axvline(-VaR_MC, color='red', linestyle='--', label=f'MC VaR 99% = ({-VaR_MC:.4f})')
plt.axvline(-ES_MC, color='teal', linestyle='--', label=f'MC ES 99% = ({-ES_MC:.4f})')
plt.axvline(-VaR_hist, color='magenta', linestyle='--', label=f'Hist VaR 99% = ({-VaR_hist:.4f})')
plt.axvline(-ES, color='purple', linestyle='--', label=f'Hist ES 99% = ({-ES:.4f})')
plt.legend(loc='upper right')
plt.savefig('figures/monte_carlo_histogram.png')
plt.show()


# scatterplot to show historical portfolio returns over time with 99% VaR threshold and breach days highlighted
plt.figure(figsize=(12, 6))
plt.plot(returns.index, portfolio_returns, color='steelblue', linewidth=0.8, label='Portfolio Returns')
plt.axhline(-VaR_hist, color='red', linestyle='--', label=f'Historical VaR 99% ({-VaR_hist:.4f})')
# highlight breach days
breaches = portfolio_returns[portfolio_returns < -VaR_hist]
plt.scatter(breaches.index, breaches, color='red', s=10, zorder=3, label='VaR Breaches')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.title('Historical Portfolio Returns with VaR Breaches (99%)')
plt.legend(loc='upper right')
plt.savefig('figures/historical_returns_var_breaches.png')
plt.show()



