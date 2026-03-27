# Portfolio Risk Analysis

This project analyses the risk of an equally weighted portfolio of major US technology stocks (AAPL, MSFT, IBM, NVDA, GOOGL, AMZN) using daily price data from 2014–2025.

## Methods
- Historical and Monte Carlo Value at Risk (VaR) and Expected Shortfall (ES) at 99% confidence
- Minimum variance portfolio optimisation using scipy
- Stress testing against the COVID-19 market crash (2020)
- GARCH(1,1) volatility modelling and dynamic VaR estimation

## Files
- `data_loader.py` — loads and processes price data from prices.xlsx
- `portfolio_risk.py` — calculates historical and Monte Carlo VaR and ES with visualisations
- `stress_test.py` — minimum variance optimisation and COVID-19 stress testing
- `garch_volatility.py` — GARCH(1,1) volatility modelling and dynamic VaR

## Key Findings
*(to be completed)*

## Requirements
- numpy, pandas, matplotlib, scipy, arch