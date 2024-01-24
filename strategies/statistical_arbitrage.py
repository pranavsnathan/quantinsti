import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations


class StatisticalArbitrage:
    def __init__(self, stock_symbols, start_date, end_date):
        """
        Initialize the class with stock symbols, start and end dates.
        """
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
    

    def download_data(self):
        """
        Downloads stock data using Yahoo Finance API.
        """
        data = {}
        for symbol in self.stock_symbols:
            try:
                data[symbol] = yf.download(symbol, start=self.start_date, end=self.end_date)['Close']
            except Exception as e:
                print(f"Couldn't download data for {symbol}. Error: {e}")
                continue

        self.dataframe = pd.DataFrame(data)
        self.dataframe.index = pd.to_datetime(self.dataframe.index)
        return self.dataframe

    
    def calculate_hedge_ratio(self):
        """
        Calculates the hedge ratio for a pair of stocks. Here the idea is to dynamically select all the possible combinations
        between all the stocks provided and then calculate the hedge ratio for each combination and store it in a dictionary.
        """
        self.hedge_ratios = {}
        self.spreads = {}
        stock_count = len(self.stock_symbols)
        for combo in combinations(self.stock_symbols, stock_count-1): 
            x = self.dataframe[list(combo)].iloc[:90]
            remaining_stock = list(set(self.stock_symbols) - set(combo))
            y = self.dataframe[remaining_stock].iloc[:90]
            model = sm.OLS(y, x).fit()
            self.hedge_ratios[combo] = model.params
            self.spreads[combo] = self.dataframe[remaining_stock].iloc[:, 0] - sum([m * self.dataframe[p] for p, m in zip(x.columns, model.params)])
        return self.spreads
    
    def plot_spreads(self):
        """
        Plots the spreads for all pairs of stocks.
        """
        for combo, spread in self.spreads.items():
            plt.figure(figsize=(10, 5))
            plt.title('Spread: {}'.format(' and '.join(combo)))
            plt.plot(spread)
            plt.ylabel('Spread')
            plt.show()

    def perform_adf_test(self):
        """
        Performs the Augmented Dickey-Fuller test to test for stationarity between all the combinations provided in the dictionary.
        This function dynamically takes the t-statistic and compares it with the critical value at 1%, 5% and 10% significance levels.
        It then prints the pair with the lowest t-statistic and returns the pair.
        """
        self.adf_results = {}
        self.cointegration_results = {}
        min_t_stat = float('inf')
        min_pair = None
        min_sig_level = None
        for perm, spread in self.spreads.items():
            adf = adfuller(spread, maxlag = 1)
            self.adf_results[perm] = adf
            for sig_level in ['1%', '5%', '10%']:  # Iterate over significance levels
                critical_value = adf[4][sig_level]
                t_stat = adf[0]
                cointegrated = t_stat < critical_value 
                if cointegrated and t_stat < min_t_stat:
                    min_t_stat = t_stat
                    min_pair = perm
                    min_sig_level = sig_level
            self.cointegration_results[perm] = cointegrated
        print(f"Chosen pair is at {min_sig_level} level of significance")
        return min_pair
    
    def stat_arb(self, min_pair, lookback, std_dev):
        """
        Performs mean reversion strategy on the pair of stocks with the lowest t-statistic.
        """
        for key, value in self.spreads.items():
            if key == min_pair:
                df = value.to_frame(name='spread')

        df['moving_average'] = df.spread.rolling(lookback).mean()
        df['moving_std_dev'] = df.spread.rolling(lookback).std()

        df['upper_band'] = df.moving_average + std_dev*df.moving_std_dev
        df['lower_band'] = df.moving_average - std_dev*df.moving_std_dev

        df['long_entry'] = df.spread < df.lower_band
        df['long_exit'] = df.spread >= df.moving_average
        df['positions_long'] = np.nan
        df.loc[df.long_entry, 'positions_long'] = 1
        df.loc[df.long_exit, 'positions_long'] = 0
        df.positions_long = df.positions_long.fillna(method='ffill')

        df['short_entry'] = df.spread > df.upper_band
        df['short_exit'] = df.spread <= df.moving_average
        df['positions_short'] = np.nan
        df.loc[df.short_entry, 'positions_short'] = -1
        df.loc[df.short_exit, 'positions_short'] = 0
        df.positions_short = df.positions_short.fillna(method='ffill')

        df['positions'] = df.positions_long + df.positions_short

        df['spread_difference'] = df.spread - df.spread.shift(1)
        df['pnl'] = df.positions.shift(1) * df.spread_difference
        df['cumpnl'] = df.pnl.cumsum()
        return df
    
    def plot_cumpnl(self, df):
        """
        Plots the PnL curve.
        """
        df.cumpnl.plot(label='Cumulative PnL', figsize=(10,7),color='magenta')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL')
        plt.show()
    