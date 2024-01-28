import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import warnings
from concurrent import futures
from itertools import product

class BigMovesMonday:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
    
    def download_data(self):
        try:
            df = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        except Exception as e:
            print(f"Couldn't download data for {symbol}. Error: {e}")
        return df
    
    def compute_daily_returns(self, df):
        """ 
        The function computes daily log returns based on the Close prices in the pandas DataFrame
        and stores it in a column  called 'cc_returns'.
        """
        df['cc_returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        return df
    
    def compute_indicators(self, df):
        """
        The function creates additional columns to an OHLC pandas DataFrame
        required to backtest the "Big Moves on Mondays" trading strategy.
        """
        # Columns created to check condition 1
        df['day'] = df.index.day_name()
        df['prev_day'] = df['day'].shift(1)
        df['four_days_after'] = df['day'].shift(-4)

        # Columns created to check condition 2
        df['relative_range'] = (df['High'] - df['Low']) / df['Adj Close']
        df['rel_range_ma'] = df['relative_range'].rolling(window=25).mean()

        # Column created to check condition 3
        df['ibs'] = (df['Adj Close'] - df['Low']) / \
            (df['High'] - df['Low'])
        return df
    
    def show_backtesting_results(self, df):
        """
        The function returns the cumulative returns from the trading strategy and a buy-and-hold strategy. 
        It also plots a chart showing both returns and position over time.
        IMPORTANT: To be run ONLY after the function backtest_strategy.
        """
        buy_and_hold_returns = np.round(df['cc_returns'].cumsum()[-1], 2)
        strategy_returns = np.round(df['strategy_returns'].cumsum()[-1], 2)

        df[['cc_returns', 'strategy_returns']] = df[[
            'cc_returns', 'strategy_returns']].cumsum()

        return {'Buy and Hold Returns': buy_and_hold_returns, 'Strategy Returns': strategy_returns}
    
    def optimize_parameters(self, df, ma_threshold_range, ibs_threshold_range):
        max_cumulative_return = 0
        best_parameters = (0, 0)

        for ma_threshold, ibs_threshold in product(ma_threshold_range, ibs_threshold_range):
            try:
                df_strategy = df.copy()
                self.backtest_strategy(df_strategy, ma_threshold, ibs_threshold)
                cumulative_return = df_strategy['strategy_returns'].sum()
                if cumulative_return > max_cumulative_return:
                    max_cumulative_return = cumulative_return
                    best_parameters = (ma_threshold, ibs_threshold)
            except Exception as e:
                print(f"An error occurred with ma_threshold {ma_threshold}, ibs_threshold {ibs_threshold}. Error: {str(e)}")

        return best_parameters
    
    def backtest_strategy(self, df, ma_threshold, ibs_threshold):
        """
        The function creates additional columns to the pandas DataFrame for checking conditions
        to backtest the "Big Moves on Mondays" trading strategy. 
        It then computes the strategy returns.
        IMPORTANT: To be run ONLY after the function compute_indicators.
        """
        # compute the daily returns and the indicators using the functions we defined
        self.compute_daily_returns(df)
        self.compute_indicators(df)

        df['condition1'] = np.where((df['day'] == 'Monday')
                                      & (df['prev_day'] == 'Friday')
                                      & (df['four_days_after'] == 'Friday'),
                                      1, 0)

        df['condition2'] = np.where((1 - df['Adj Close'] / df['Adj Close'].shift(1))
                                      >= ma_threshold * df['rel_range_ma'], 1, 0)

        df['condition3'] = np.where(df['ibs'] < ibs_threshold, 1, 0)

        df['signal'] = np.where((df['condition1'] == 1)
                                  & (df['condition2'] == 1)
                                  & (df['condition3'] == 1),
                                  1, 0)

        # The below two statements ensures that we can directly calculate strategy returns by multiplying the
        # columns 'position' and 'cc_returns'
        df['signal'] = df['signal'].shift(1)

        df['position'] = df['signal'].replace(
            to_replace=0, method='ffill', limit=3)

        df['strategy_returns'] = df['cc_returns'] * df['position']

        # print the results
        self.show_backtesting_results(df)