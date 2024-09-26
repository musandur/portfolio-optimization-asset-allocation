import pandas as pd
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

def get_data(filename):
    '''
    Function opens a file and extracts the dataset

    INPUTS:
        filename: directory to the dataset file
    RETURN:
        multi-index pandas DataFrame.
    '''

    data = pd.read_csv(filename, index_col=[0, 1], parse_dates=True)
    data = data.loc[pd.IndexSlice['2019-08-08':, :], :]

    return data


def  model_alpha_factor(df, past_days):
    
    '''
    Function computes the alpha factors as defined

    INPUTS
        df: dataset
        past_days:  backward window length considered to compute max highs and min lows

    RETURN: Multi-Index DataFrame of  3 columns [Adj Close, max_highs, min_lows] 
            and a single column Multi-Index DataFrame [alpha factor]


    '''
    
    close = df['Adj Close']  
    high = df['High'].unstack()
    low = df['Low'].unstack()
    
    min_lows = low.shift(1).rolling(past_days).min().dropna()
    max_highs = high.shift(1).rolling(past_days).max().dropna()

    df_lows = pd.DataFrame(min_lows.stack(), columns=['lowest_lows'])
    df_highs = pd.DataFrame(max_highs.stack(), columns=['highest_highs'])
    
    df_closes = close.to_frame() 
    df_closes_ret = df_closes.pct_change(past_days).dropna()
    df_closes_ret.rename(columns={'Adj Close': 'past_days_ret'}, inplace=True)

    list_dfs = [df_closes, df_highs, df_lows, df_closes_ret]
    result_df = reduce(lambda left, right: pd.merge(left, 
                                                    right, 
                                                    on=['Date', 'Ticker'], 
                                                    how='inner'), list_dfs)
                                                    
    
    result_df['alpha_factor'] = result_df.apply(lambda row: row['Adj Close'] - row['lowest_lows'] 
                                         if row['Adj Close'] < row['lowest_lows'] else(
                                             row['Adj Close'] - row['highest_highs'] 
                                             if row['Adj Close'] > row['highest_highs'] else 
                                                 row['past_days_ret']*0.001
                                         ), axis=1)

    return result_df, result_df[['alpha_factor']]

if __name__ == '__main__':
    print(get_data('breakout.csv'))