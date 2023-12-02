#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:42:51 2023

@author: yunxiyang
"""

# Exploratory Data Analysis(EDA) on Bitcoin and Stock Data

# import necessary libraries
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 


# import the dataset
stock = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Processed_Data/stock_combined.csv')

"""
Data Summary
"""
# Generate descriptive statistics
desc = stock.describe()

# Select only the desired rows and round to 2 decimal places
desc = desc.loc[['count', 'mean', '50%', 'max', 'min']].round(2)
desc.to_csv("stock_data_summary")


"""
Data Visulization
"""
# Trend of Close Price
fig = plt.figure(figsize = (15,10))
fig.suptitle("Date Range: 2021.02.01 - 2022.01.31", fontsize=16)
fig.tight_layout()

# Plot for trend of Bitcoin close price
plt.subplot(2, 3, 1)
plt.plot(stock['date'], stock['btc_close'], color="red")
plt.title('Bitcoin Close Price')

# Plot for trend of Nasdaq Composite Close Price
plt.subplot(2, 3, 2)
plt.plot(stock['date'], stock['nasdaq_close'],color="blue")
plt.title('Nasdaq Composite Close Price')

# Plot for trend of Dow Jones Index Average Close Price
plt.subplot(2, 3, 3)
plt.plot(stock['date'], stock['dji_close'],color="grey")
plt.title('Dow Jones Index Average Close Price')

# Plot for trend of S&P 500 Close Price
plt.subplot(2, 3, 4)
plt.plot(stock['date'], stock['sp_close'],color="brown")
plt.title('S&P 500 Close Price)')

# Plot for trend of Crude Oil Close Price
plt.subplot(2, 3, 5)
plt.plot(stock['date'], stock['oil_close'],color="black")
plt.title('Crude Oil Close Price)')

# Plot for trend of Gold Close Price
plt.subplot(2, 3, 6)
plt.plot(stock['date'], stock['gold_close'],color="purple")
plt.title('Gold Close Price)')
# Save the figure as a JPG file
fig.savefig('close_price_trend.jpg', dpi=600)

# Plot for trend of Bitcoin stock volume
fig = plt.figure(figsize = (15,10))
plt.plot(stock['date'] , stock['btc_volume'])
plt.title('Volume of Bitcoin')
# Save the figure as a JPG file
fig.savefig('bitcoin_volume.jpg', dpi=300)

# Plot for comparison of close and open prices
fig = plt.figure(figsize = (15,10))
plt.plot(stock['date'], stock['btc_close'])
plt.plot(stock['date'], stock['btc_open'])
plt.legend(["C", "O"])
plt.title('Comparision of close and open prices of Bitcoin')
# Save the figure as a JPG file
fig.savefig('comparision_of_bitcoin_close_and_open_prices.jpg', dpi=300)

# Plot for trend of moving average
# As we know the stock prices are highly volatile and prices change quickly with time. 
# To observe any trend or pattern we can take the help of a 50-day 200-day average.
fig = plt.figure(figsize = (15,10))
plt.plot(stock['date'], stock['btc_close'].rolling(10).mean())
plt.plot(stock['date'], stock['btc_close'].rolling(20).mean())
plt.plot(stock['date'], stock['btc_close'].rolling(50).mean())
plt.title('Bitcoin Close Price moving average')
# Save the figure as a JPG file
fig.savefig('bitcoin_close_price_moving_average.jpg', dpi=300)

# Histogram with mean indicator of Bitcoin
fig = plt.figure(figsize = (20,10))
sns.histplot(stock['btc_close'],color='darkred', kde=True, binwidth=500)
plt.axvline(stock['btc_close'].mean(), color='k', linestyle='dashed', linewidth=3)
plt.title('Bitcoin Close Price')
# Save the figure as a JPG file
fig.savefig('bitcoin_close_price.jpg', dpi=300)


# Plot for Bitcoin Daily Return
fig = plt.figure(figsize = (15,10))
plt.plot(stock['date'], stock['btc_close'].pct_change())
plt.title('Bitcoin Daily Return')
# Save the figure as a JPG file
fig.savefig('bitcoin_daily_return.jpg', dpi=300)
















