import argparse
import numpy as np
import pandas as pd
from strategies.statistical_arbitrage import StatisticalArbitrage
from strategies.big_moves_monday import BigMovesMonday 

def main(strategy):
    if strategy == 'stat_arb':
        stocks_list = ['GLD', 'GDX', 'USO']
        start_date = '2014-12-31'
        end_date = '2023-01-28'
        stat_arb = StatisticalArbitrage(stocks_list, start_date, end_date)
        data = stat_arb.download_data()
        spreads = stat_arb.calculate_hedge_ratio()
        chosen_pair = stat_arb.perform_adf_test()
        results = stat_arb.stat_arb(chosen_pair, 15, 1)
        stat_arb.plot_cumpnl(results)
    elif strategy == 'big_moves_monday':
        results = []
        stock_symbols = ['SPY', 'QQQ', 'IWM', 'EEM']
        start_date = '2017-01-01'
        end_date = '2024-01-27'
        # Define the ranges for the parameters
        ma_threshold_range = np.arange(0.1, 1, 0.01)
        ibs_threshold_range = np.arange(0.1, 1, 0.01) 
        for stock_symbol in stock_symbols:
            try:
                print(f"Processing {stock_symbol}")
                bbm = BigMovesMonday(stock_symbol, start_date, end_date)
                df = bbm.download_data()

                # Optimize parameters
                best_ma_threshold, best_ibs_threshold = bbm.optimize_parameters(df, ma_threshold_range, ibs_threshold_range)
                print(f"Best parameters for {stock_symbol}: ma_threshold={best_ma_threshold}, ibs_threshold={best_ibs_threshold}")

                # Backtest strategy using the best parameters
                bbm.backtest_strategy(df=df, ma_threshold=best_ma_threshold, ibs_threshold=best_ibs_threshold)
                
                result = bbm.show_backtesting_results(df)
                result['Stock'] = stock_symbol
                result['ma_threshold'] = best_ma_threshold
                result['ibs_threshold'] = best_ibs_threshold
                results.append(result)
            except Exception as e:
                print(f"An error occurred while processing {stock_symbol}. Error: {str(e)}")
            print("\n")

        # Convert results to DataFrame for better visualization
        results_df = pd.DataFrame(results)
        print(results_df)
    else:
        print(f"Unknown strategy: {strategy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('strategy', type=str, help="The strategy to run. Options are 'stat_arb' or 'big_moves_monday'")
    args = parser.parse_args()
    main(args.strategy)