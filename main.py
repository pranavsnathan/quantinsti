from strategies.statistical_arbitrage import StatisticalArbitrage

def main():
    stocks_list = ['GLD', 'GDX', 'USO']
    start_date = '2014-12-31'
    end_date = '2023-01-28'
    stat_arb = StatisticalArbitrage(stocks_list, start_date, end_date)
    data = stat_arb.download_data()
    spreads = stat_arb.calculate_hedge_ratio()
    chosen_pair = stat_arb.perform_adf_test()
    results = stat_arb.stat_arb(chosen_pair, 15, 1)
    stat_arb.plot_cumpnl(results)


if __name__ == '__main__':
    main()