#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:41:35 2023

@author: zozochunyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:03:53 2023

@author: zozo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

outpath = "/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output"

# import dataset
data = pd.read_csv('/Users/zozochunyu/Biostats_Fall23/QMSS5067NLP/Final/output/stock_polarity_ver2.csv',
                   parse_dates = True, index_col = 0)
# Calculate the return rate
data['return'] = data['btc_close'].pct_change()
data['return_1'] = data['price_1d'].pct_change()
data['return_2'] = data['price_2d'].pct_change()
data['return_3'] = data['price_3d'].pct_change()

data['nasdaq_1d'] = data['nasdaq_1d'].pct_change()
data['nasdaq_2d'] = data['nasdaq_2d'].pct_change()
data['nasdaq_3d'] = data['nasdaq_3d'].pct_change()

data['sp_1d'] = data['sp_1d'].pct_change()
data['sp_2d'] = data['sp_2d'].pct_change()
data['sp_3d'] = data['sp_3d'].pct_change()


# data['dji_1d'] = data['dji_1d'].pct_change()
# data['dji_2d'] = data['dji_2d'].pct_change()
# data['dji_3d'] = data['dji_3d'].pct_change()
# data.to_csv(outpath+'/return_polarity.csv')
data.dropna(inplace=True)
data.drop("date",axis=1,inplace=True)

print(data.head(5))



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



"""
Selecting features and target

"""
# baseline
x = data[['nasdaq_1d','nasdaq_2d', 'nasdaq_3d','sp_1d', 'sp_2d', 'sp_3d', 
          # 'btc_open' highly correlated
          ]]
y = data['return']

# situation 1: add polarity score as one of the predictors
x_1 = data[['nasdaq_1d','nasdaq_2d', 'nasdaq_3d','sp_1d', 'sp_2d', 'sp_3d', 
            'compound'
          ]]
# y = data['return']


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


# Random Forest
# Defining the parameter grid for tuning the Random Forest Regressor
param_grid_rf = {
    'n_estimators': [100, 500, 1000],
    'max_depth':[None, 5, 10],
    'min_samples_split': [5, 10]
}

# Initializing GridSearchCV for the baseline model (Random Forest)
grid_search_rf_baseline = GridSearchCV(RandomForestRegressor(random_state=0), param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf_baseline.fit(x_train, y_train)
# Best parameters and score for the baseline model (Random Forest)
best_params_rf_baseline = grid_search_rf_baseline.best_params_
rf_model = RandomForestRegressor(**best_params_rf_baseline, random_state=0)
rf_model.fit(x_train, y_train)

# Initializing GridSearchCV for situation 1 model (Random Forest)
grid_search_rf_situation1 = GridSearchCV(RandomForestRegressor(random_state=0), param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf_situation1.fit(x_train1, y_train1)

# Best parameters and score for situation 1 model (Random Forest)
best_params_rf_situation1 = grid_search_rf_situation1.best_params_
rf_model1 = RandomForestRegressor(**best_params_rf_situation1, random_state=0)
rf_model1.fit(x_train1, y_train1)



"""
Prediction models - Testing (making prediction)

"""
# Linear Regression
linear_pred = linear_model.predict(x_test) # -------- baseline
linear_pred1 = linear_model1.predict(x_test1) # -------- situation 1


# Random Forest Regression
rf_pred = rf_model.predict(x_test)
rf_pred1 = rf_model1.predict(x_test1)



# 
"""
Model Evaluation - test RMSE, R2 Score

"""

# Creating a DataFrame to print out the results
model_results = {
    ('Linear Regression', 'Baseline'): {'RMSE': np.sqrt(mean_squared_error(y_test, linear_pred)), 
                                        'R2 Score': r2_score(y_test, linear_pred)},
    ('Linear Regression', 'Situation1'): {'RMSE': np.sqrt(mean_squared_error(y_test1, linear_pred1)), 
                                          'R2 Score': r2_score(y_test1, linear_pred1)},
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

