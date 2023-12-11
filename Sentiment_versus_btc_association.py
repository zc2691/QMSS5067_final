#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:16:20 2023

@author: yunxiyang
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns 

"""
Association Exploration
"""
outpath = "/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/"

# import data
data = pd.read_csv("/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Processed_Data/full_data.csv")

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

"""
Polarity Score v.s. Bitcoin Closing Price
"""

# Plotting
plt.figure(figsize=(20, 6))
plt.plot(df.index, df['btc_close'], color='blue', label='BTC Closing Price')
plt.xlabel('Date')
plt.ylabel('BTC Closing Price', color='blue')
plt.twinx()
plt.plot(df.index, df['compound'], color='green', label='Polarity Score')
plt.ylabel('Polarity Score', color='green')
plt.title('Polarity Score vs BTC Closing Price Over Time')
plt.legend(loc='upper left')
# Save the figure as a PNG file
plt.savefig('Polarity Score vs BTC Closing Price Over Time.png', dpi=300)
plt.show()


# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['compound'], df['btc_close'])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df['compound'], df['btc_close'], label='Data points')
plt.plot(df['compound'], intercept + slope * df['compound'], color='red', label=f'Linear Best-fit (Gradient: {slope:.2f})')
plt.title('Association between Polarity Compound Score and Bitcoin Closing Price')
plt.xlabel('Polarity Score')
plt.ylabel('Bitcoin Closing Price (USD)')
plt.legend()
plt.grid(True)
# Save the figure as a PNG file
plt.savefig('Association between Polarity Compound Score and Bitcoin Closing Price.png', dpi=300)
plt.show()

"""
NB Class Label v.s. Bitcoin Closing Price
"""
nb_data = pd.read_csv("/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Processed_Data/stock_combined_nb.csv", index_col=0)

nb_df = pd.DataFrame(nb_data)
nb_df['date'] = pd.to_datetime(nb_df['date'])

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment_bayes', y='btc_close', data=nb_df)
plt.title('Bitcoin Closing Prices by Naive Bayes Classification Outcome')
plt.xlabel('Naive Bayes Classification (Sentiment)')
plt.ylabel('Bitcoin Closing Price (USD)')
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('btc_prices_by_nb_classification.png', dpi=300)
# Show the plot
plt.show()