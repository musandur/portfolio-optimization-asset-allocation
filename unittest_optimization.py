import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import date, timedelta

import optimization_module


class TestPortfolioOptimization(unittest.TestCase):
    '''
        unit test class for testing the optimization module "optimization.py".
        Especially, it test the fonction "get_optimal_weights()" that uses all the functions
        class of the module "optimization.py".

        RUN: python test_optimization.py or pytest optimization.py
    '''

    def generate_random_tickers(self, num_tickers=None):
            
        min_ticker_len = 2
        max_ticker_len = 4
        if not num_tickers:
            num_tickers = np.random.randint(6, 8)
        ticker_symbol_random = np.random.randint(ord('A'), ord('Z')+1, (num_tickers, max_ticker_len))
        ticker_symbol_lengths = np.random.randint(min_ticker_len, max_ticker_len, num_tickers)

        tickers = []  # list of tickers
        for ticker_symbol_rand, ticker_symbol_length in zip(ticker_symbol_random, ticker_symbol_lengths):
            ticker_symbol = ''.join([chr(c_id) for c_id in ticker_symbol_rand[:ticker_symbol_length]])
            tickers.append(ticker_symbol)

        return tickers
        
    def generate_random_dates(self, num_dates=None):

        if not num_dates:
            num_dates = np.random.randint(4, 7)
        
        start_year = np.random.randint(2019, 2023)
        start_month = np.random.randint(1, 12)
        start_day = np.random.randint(1, 29)
        start_date = date(start_year, start_month, start_day)

        dates = []  # list of dates
        for i in range(num_dates):
            dates.append(start_date + timedelta(days=i))

        return dates
        
    def generate_multi_index_df(self, tickers, dates, columns_names, index_names):

        index = pd.MultiIndex.from_product([tickers, dates], names=index_names)
    
        data = 2*np.random.rand(len(index), len(columns_names)) - 1

        return pd.DataFrame(data, index=index, columns=columns_names)

    @patch('optimization_module.OptimalHoldings')
    @patch('optimization_module.factor_betas_and_specific_return')
    def test_get_optimal_weights(self, mock_betas_and_specific_return, mock_optimal_holdings):        
    
        factors = ['F1', 'F2', 'F3', 'F4', 'F5']
        num_factors = len(factors)
        tickers = self.generate_random_tickers(7)
        dates = self.generate_random_dates(4)
        num_tickers = len(tickers)
        num_dates = len(dates)

        betas = pd.DataFrame(2*np.random.rand(num_tickers, num_factors)-1, columns=factors, index=tickers)
        betas[factors[0]] = np.abs(betas[factors[0]])
        print('Betas')
        print(betas)
        specific_ret = pd.DataFrame(2*np.random.rand(num_dates, num_tickers)-1, columns=tickers, index=dates)
        mock_betas_and_specific_return.return_value = (betas, specific_ret)

        mock_instance = MagicMock()
        mock_optimal_holdings.return_value = mock_instance
        mock_instance.find_optimal_holdings.return_value = \
            pd.DataFrame({'optimal_weights': 2*np.random.rand(num_tickers) - 1}, index=tickers)
                         
        f_data = 2*np.random.rand(num_dates, num_factors) - 1
        factor_data = pd.DataFrame(f_data, index=dates, columns=factors)
        factor_data.index.name = 'date'
        asset_ret = self.generate_multi_index_df(tickers, 
                                                dates, 
                                                columns_names=['return'], 
                                                index_names=['Ticker', 'date'])
        
        fama_fac_and_return = factor_data.join(asset_ret)

        a_data = 2*np.random.rand(num_tickers) - 1
        alpha_vector = pd.DataFrame({'alpha': a_data}, index=tickers)

        print("Alpha vector:")
        print(alpha_vector)

        print("Fama-French and return data:")
        print(fama_fac_and_return)

        # Call the function with the mocked data
        result = optimization_module.get_optimal_weights(alpha_vector, fama_fac_and_return, factor_data)

        print("Result:")
        print(result)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(alpha_vector))
        self.assertIn('optimal_weights', result.columns)

        mock_betas_and_specific_return.assert_called_once_with(fama_fac_and_return, factor_data)
        mock_instance.find_optimal_holdings.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)


