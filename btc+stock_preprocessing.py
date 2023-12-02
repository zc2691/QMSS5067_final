#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:35:29 2023

@author: yunxiyang
"""

import pandas as pd

# Read the datasets
btc_raw = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Raw_Data/BitCoin_USD_2014-2022.csv')
nasdaq_raw = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Raw_Data/nasdaq_raw.csv')
dji_raw = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Raw_Data/dji_raw.csv')
sp_raw = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Raw_Data/sp_raw.csv')
oil_raw = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Raw_Data/oil_raw.csv')
gold_raw = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Raw_Data/gold_raw.csv')

# rename the chosen column in each dataframe
start_date = '2021-02-01'
end_date = '2022-01-31'

btc = btc_raw[(btc_raw['Date'] >= start_date) & (btc_raw['Date'] <= end_date)].rename(columns={'Close': 'btc_close', 'Open': 'btc_open', 'Volume': 'btc_volume'})
nasdaq = nasdaq_raw.rename(columns={'Close': 'nasdaq_close'})
dji = dji_raw.rename(columns={'Close': 'dji_close'})
sp = sp_raw.rename(columns={'Close': 'sp_close'})
oil = oil_raw.rename(columns={'Close': 'oil_close'})
gold = gold_raw.rename(columns={'Close': 'gold_close'})

# Merge the datasets on 'date' to form a dataset 
stock_combined = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(
                    btc[['Date','btc_open','btc_close','btc_volume']], 
                    nasdaq[['Date', 'nasdaq_close']], on='Date', how='left'), 
                    dji[['Date', 'dji_close']], on='Date', how='left'),
                    sp[['Date', 'sp_close']], on='Date', how='left'),
                    oil[['Date', 'oil_close']], on='Date', how='left'),
                    gold[['Date', 'gold_close']], on='Date', how='left').rename(columns={'Date': 'date'})

# Display the filtered stock dataset
print(stock_combined)

# save the output dataframe as csv
stock_combined.to_csv('stock_combined.csv', index=False)

