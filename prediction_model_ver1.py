#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:03:53 2023

@author: yunxiyang
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

outpath = "/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject"

# import dataset
data = pd.read_csv('/Users/yunxiyang/Desktop/nlp/yy3297_gr5067_FinalProject/Processed_Data/full_combined+polarity.csv',
                   parse_dates = True, index_col = 0)
# Calculate the return rate
data['btc_return'] = ((data['btc_close'] - data['btc_open']) / data['btc_open']) * 100


"""
Correlation matrix for feature selection

"""
corr = data.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(550, 20, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

ax.set_title('Feature Correlation Matrix')
plt.show()



# Generate a custom diverging colormap
cmap = sns.diverging_palette(550, 20, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

ax.set_title('Feature Correlation Matrix')
plt.show()

"""
Selecting features and target

"""
# baseline
x = data[['btc_volume', 'nasdaq_close', 'sp_close', 
          'dji_close', 'oil_close', 'gold_close'
          # 'btc_open' highly correlated
          ]]
y = data['btc_return']

# situation 1: add polarity score as one of the predictors
x_1 = data[['btc_volume', 'nasdaq_close', 'sp_close', 
            'dji_close', 'oil_close', 'gold_close', 'compound'
          # 'btc_open'&'btc close', 'postive'&'neutral', highly correlated
          ]]
# y = data['btc_return']


""" 
Data partition

"""
# baseline
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
print('Data Split for baseline done.')

# situation 1: add polarity score as one of the predictors
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_1, y, test_size=0.20, random_state=0)
print('Data Split for situation 1 done.')


"""
Prediction models - Training

"""
# Linear Regression

linear_model = LinearRegression()
linear_model.fit(x_train, y_train) # -------- baseline

linear_model1 = LinearRegression()
linear_model1.fit(x_train1, y_train1) # -------- situation 1

# Lasso Regression

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train) # -------- baseline

lasso_model1 = Lasso(alpha=0.1)
lasso_model1.fit(x_train1, y_train1) # -------- situation 1

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators = 5)
rf_model.fit(x_train, y_train) # -------- baseline

rf_model1 = RandomForestRegressor(n_estimators = 5)
rf_model1.fit(x_train1, y_train1) # -------- situation 1


"""
Prediction models - Testing (making prediction)

"""
# Linear Regression
linear_pred = linear_model.predict(x_test) # -------- baseline
linear_pred1 = linear_model1.predict(x_test1) # -------- situation 1

# Lasso Regression
lasso_pred = lasso_model.predict(x_test) # -------- baseline
lasso_pred1 = lasso_model1.predict(x_test1) # -------- situation 1

# Random Forest Regression
rf_pred = rf_model.predict(x_test) # -------- baseline
rf_pred1 = rf_model1.predict(x_test1) # -------- situation 1


"""
Model Evaluation - test RMSE, R2 Score

"""

# Creating a DataFrame to print out the results
model_results = {
    ('Linear Regression', 'Baseline'): {'RMSE': np.sqrt(mean_squared_error(y_test, linear_pred)), 
                                        'R2 Score': r2_score(y_test, linear_pred)},
    ('Linear Regression', 'Situation1'): {'RMSE': np.sqrt(mean_squared_error(y_test1, linear_pred1)), 
                                          'R2 Score': r2_score(y_test1, linear_pred1)},
    ('Lasso Regression', 'Baseline'): {'RMSE': np.sqrt(mean_squared_error(y_test, lasso_pred)), 
                                       'R2 Score': r2_score(y_test, lasso_pred)},
    ('Lasso Regression', 'Situation1'): {'RMSE': np.sqrt(mean_squared_error(y_test1, lasso_pred1)), 
                                         'R2 Score': r2_score(y_test1, lasso_pred1)},
    ('Random Forest', 'Baseline'): {'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)), 
                                       'R2 Score': r2_score(y_test, rf_pred)},
    ('Random Forest', 'Situation1'): {'RMSE': np.sqrt(mean_squared_error(y_test1, rf_pred1)), 
                                         'R2 Score': r2_score(y_test1, rf_pred1)}
}

# Create a MultiIndex DataFrame
model_results_df = pd.DataFrame(model_results)

# Optional: Round the values for readability
model_results_df = model_results_df.round(5)

print(model_results_df)