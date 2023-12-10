#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:28:25 2023

@author: zozochunyu
"""

import pandas as pd

stock_data = pd.read_csv(
    '/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/data/stock_combined.csv')
outpath = "/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output"
directory = '/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final'


# fill in na values
stock_data= stock_data.fillna(method='backfill')

# Price_1d, Price_2d, Price_3d
stock_data["price_1d"] = stock_data.btc_close.shift(1)
stock_data["price_2d"] = stock_data.btc_close.shift(2)
stock_data["price_3d"] = stock_data.btc_close.shift(3)

stock_data["nasdaq_1d"] = stock_data.nasdaq_close.shift(1)
stock_data["nasdaq_2d"] = stock_data.nasdaq_close.shift(2)
stock_data["nasdaq_3d"] = stock_data.nasdaq_close.shift(3)

stock_data["sp_1d"] = stock_data.sp_close.shift(1)
stock_data["sp_2d"] = stock_data.sp_close.shift(2)
stock_data["sp_3d"] = stock_data.sp_close.shift(3)

stock_data["dji_1d"] = stock_data.dji_close.shift(1)
stock_data["dji_2d"] = stock_data.dji_close.shift(2)
stock_data["dji_3d"] = stock_data.dji_close.shift(3)

stock_data.to_csv('/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output/stock_data_ver2.csv') 


import re
import pandas as pd 
import numpy as np
# combining tweets and stock_data_ver2 datasets
##### combine stock data and sentiment scores from grouped tweets every day
stock = pd.read_csv(directory+'/output/stock_data_ver2.csv',index_col=0)
sentiment_scores_comb = pd.read_csv(directory+'/output/sentiment_scores_comb.csv', index_col=0)

stock_polarity_ver2 = stock_data.merge(sentiment_scores_comb, on = 'date', how='inner').dropna(axis=0,how='any')
stock_polarity_ver2 = stock_polarity_ver2.dropna(axis=0,how='any')
stock_polarity_ver2.columns
stock_polarity_ver2.to_csv(directory+"/output/stock_polarity_ver2.csv")

