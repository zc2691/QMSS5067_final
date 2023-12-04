#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:40:01 2023

@author: zozochunyu
"""

import os
directory = '/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final'
os.chdir(directory)
# Import Data Preprocessing and Wrangling libraries
import re
import pandas as pd 
import numpy as np
# combining tweets and stock datasets
##### combine stock data and sentiment scores from grouped tweets every day
stock = pd.read_csv(directory+'/data/stock_combined.csv',index_col=0)
sentiment_scores_comb = pd.read_csv(directory+'/output/sentiment_scores_comb.csv', index_col=0)

full_data = stock.merge(sentiment_scores_comb, on = 'date', how='left').dropna(axis=0,how='any')
full_data = full_data.dropna(axis=0,how='any')
full_data.columns
full_data.to_csv(directory+"/output/full_data.csv")

