# Portfolio Risk Analysis
This project analyses the risk of an equally weighted portfolio of major US technology stocks (AAPL, MSFT, IBM, NVDA, GOOGL, AMZN) using daily price data from 2014–2025.

## Methods
- Historical and Monte Carlo Value at Risk (VaR) and Expected Shortfall (ES) at 99% confidence
- Minimum variance portfolio optimisation using scipy
- Stress testing against the COVID-19 market crash (2020)
- GARCH(1,1) volatility modelling and dynamic VaR estimation
- Backtesting using Kupiec test, Ljung-Box test and Basel traffic light framework

## Files
- `data_loader.py` — loads and processes price data from prices.xlsx
- `portfolio_risk.py` — calculates historical and Monte Carlo VaR and ES with visualisations
- `stress_test.py` — minimum variance optimisation and COVID-19 stress testing
- `garch_volatility.py` — GARCH(1,1) volatility modelling and dynamic VaR
- `backtesting.py` — formal VaR model validation using statistical tests and regulatory framework

## Key Findings
- Historical VaR (99%) was 4.37% and ES was 5.64%, with Monte Carlo estimates lower at 3.4% and 3.9% respectively, reflecting the normality assumption underlying Monte Carlo simulation
- COVID-19 stress testing revealed crisis VaR of 6.72% and ES of 10.21% — approximately 54% and 81% higher than normal period estimates
- Minimum variance optimisation produced equal weights across all six assets, reflecting the high correlation structure of the tech sector
- GARCH(1,1) conditional volatility spiked to 6.6% during the COVID crash compared to roughly 1% in calm periods, demonstrating the failure of constant volatility assumptions
- Backtesting confirmed the correct breach rate (30 breaches, Kupiec p-value 0.9956) but the Ljung-Box test revealed significant breach clustering around March 2020, a known limitation of static historical VaR

## Requirements
- numpy, pandas, matplotlib, scipy, arch, statsmodels