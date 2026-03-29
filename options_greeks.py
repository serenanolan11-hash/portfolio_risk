import numpy as np
import pandas as pd
from scipy.stats import norm

# Black-Scholes model to price European call and put options and calculate Greeks
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
    theta_put = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = - K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return call_price, put_price, delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put

# AAPL option parameters
S = 227.0     
K = 230.0      
T = 0.25      
r = 0.05    
sigma = 0.018 * np.sqrt(252)  # annualised volatility from data

call_price, put_price, delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put = black_scholes(S, K, T, r, sigma)

results = pd.DataFrame({
    'Price': [call_price, put_price],
    'Delta': [delta_call, delta_put],
    'Gamma': [gamma, gamma],
    'Vega': [vega, vega],
    'Theta': [theta_call, theta_put],
    'Rho': [rho_call, rho_put]
}, index=['Call', 'Put'])
print(results)

import matplotlib.pyplot as plt
stock_prices = range(150, 300)

# plot delta across a range of stock prices to show sensitivity to underlying price
deltas_call = []
deltas_put = []
for s in stock_prices:
    result = black_scholes(s, K, T, r, sigma)
    deltas_call.append(result[2])
    deltas_put.append(result[3])
plt.figure(figsize=(12, 6))
plt.plot(stock_prices, deltas_call, color='blue', label='Call Delta')
plt.plot(stock_prices, deltas_put, color='red', label='Put Delta')
plt.axvline(S, color='black', linestyle='--', label=f'Current Price (${S})')
plt.axhline(0, color='grey', linewidth=0.5)
plt.xlabel('Stock Price')
plt.ylabel('Delta')
plt.title('Option Delta vs Stock Price (AAPL)')
plt.legend(loc='upper left')
plt.savefig('figures/delta_vs_price.png')
plt.show()



# plot gamma to show where delta changes fastest
gamma = []
for s in stock_prices:
    result = black_scholes(s, K, T, r, sigma)
    gamma.append(result[4])
plt.figure(figsize=(12, 6))
plt.plot(stock_prices, gamma, color='blue', label='Gamma')
plt.axvline(S, color='black', linestyle='--', label=f'Current Price (${S})')
plt.axhline(0, color='grey', linewidth=0.5)
plt.xlabel('Stock Price')
plt.ylabel('Gamma')
plt.title('Option Gamma vs Stock Price (AAPL)')
plt.legend(loc='upper left')
plt.savefig('figures/gamma_vs_price.png')
plt.show()



# simulate stock price path over 63 trading days
np.random.seed(42)  # for reproducibility
n_days = 63
dt = 1/252
simulated_prices = [S]

for i in range(n_days):
    daily_return = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    simulated_prices.append(simulated_prices[-1] * daily_return)

    # delta hedging simulation - rebalance hedge daily
hedge_position = []
option_value = []
time_remaining = []

for i in range(n_days + 1):
    t_remaining = (n_days - i) / 252
    if t_remaining == 0:
        t_remaining = 1/252  # avoid division by zero on expiry
    
    result = black_scholes(simulated_prices[i], K, t_remaining, r, sigma)
    hedge_position.append(result[2])  # delta = shares to hold
    option_value.append(result[0])    # call price
    time_remaining.append(t_remaining)

    # plot simulated stock price path and delta hedge ratio over time
import matplotlib.pyplot as plt
days = range(n_days + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# top plot - simulated stock price
ax1.plot(days, simulated_prices, color='steelblue', label='Simulated AAPL Price')
ax1.axhline(K, color='red', linestyle='--', label=f'Strike Price (${K})')
ax1.set_ylabel('Stock Price')
ax1.set_title('Delta Hedging Simulation (AAPL, 3 Month Call Option)')
ax1.legend(loc='upper right')

# bottom plot - delta over time
ax2.plot(days, hedge_position, color='orange', label='Delta (Hedge Ratio)')
ax2.set_ylabel('Delta')
ax2.set_xlabel('Trading Days')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/delta_hedging.png')
plt.show()