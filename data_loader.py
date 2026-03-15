import pandas as pd
import numpy as np

def load_data():

    # load stock price data from Excel
    prices = pd.read_excel("prices.xlsx", index_col=0)

    # convert the index to datetime format
    prices.index = pd.to_datetime(prices.index)

    # compute daily log returns from price data
    returns = np.log(prices / prices.shift(1)).dropna()

    return prices, returns
