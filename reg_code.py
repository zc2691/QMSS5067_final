# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:15:22 2023

@author: illge
"""

"""
1. OLS on all data 

create training data and testing data 
training data: 80% (188 rows)
testing data: 20% (47 rows)
"""

import pandas as pd
path = "C:/Users/illge/Downloads/full_data.csv"
all_data = pd.read_csv(path)

#separate training and testing data
train_data = all_data[:187]
test_data = all_data[187:]

"""
run OLS
"""

from sklearn.linear_model import LinearRegression

#create Y and X for regression
Xtr = train_data[['compound', 'btc_volume', 'nasdaq_close', 'dji_close', 'sp_close', 'oil_close', 'gold_close']]
Ytr = train_data['btc_close']

model = LinearRegression()

from sklearn.preprocessing import StandardScaler
# "Standardize"
scaler = StandardScaler()
X_scaledOLS = scaler.fit_transform(Xtr)

model.fit(X_scaledOLS, Ytr)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

#stat sig: OLS
from scipy import stats
import numpy as np 

#write function to calculate standard error
def secalc(X, Y, model):
    X1 = np.column_stack([np.ones((X.shape[0], 1)), X])
    XprimeXinv = np.linalg.inv(X1.T @ X1)

    e = Y - model.predict(X)

    var = (XprimeXinv @ X1.T @ np.diag(e**2) @ X1 @ XprimeXinv)
    std_error = var[1, 1]**0.5
    t_stat = model.coef_[0] / std_error
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(Y)-7))
    return print(f'Test Statistic: {t_stat}', f'P-value: {p_value}', f'Est Std. Error: {std_error}')
 
secalc(X_scaledOLS, Ytr, model)
   
#create predictions 
X_test_ols_all = scaler.fit_transform( 
    test_data[['compound', 'btc_volume', 'nasdaq_close', 'dji_close', 'sp_close', 'oil_close', 'gold_close']].values
    )

predicted_values = model.predict(X_test_ols_all)

"""
2. OLS model with only crypto quotes 
"""

path1 = "C:/Users/illge/Downloads/full_data_filtered.csv"
crypto_data = pd.read_csv(path1)

#separate training and testing data
ctrain_data = crypto_data[:56]
ctest_data = crypto_data[56:]

#create X and Y
Xsc_crypto = scaler.fit_transform( 
    ctrain_data[['compound', 'btc_volume', 'nasdaq_close', 'dji_close', 'sp_close', 'oil_close', 'gold_close']]
    )
Ycrypto = ctrain_data['btc_close']

cols_model = LinearRegression()

cols_model.fit(Xsc_crypto, Ycrypto)

print('Coefficients:', cols_model.coef_)
print('Intercept:', cols_model.intercept_)

#stat sig: 
secalc(Xsc_crypto, Ycrypto, cols_model)
    
#create predictions 
X_test_ols_crypto = scaler.fit_transform( 
    ctest_data[['compound', 'btc_volume', 'nasdaq_close', 'dji_close', 'sp_close', 'oil_close', 'gold_close']].values
    )

predicted_values2 = cols_model.predict(X_test_ols_crypto)

"""
3. LASSO model with (k=) 5-fold cross validation, all data
"""

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold

lasso_model = Lasso()

# Define the range of \alpha values to search
alphas = [0, 0.0001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

# k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'alpha': alphas}
grid_search = GridSearchCV(lasso_model, param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X_scaledOLS, Ytr)

best_alpha = grid_search.best_params_['alpha']
# print best value of \alpha
print('Best alpha:', best_alpha)

best_lasso_model = grid_search.best_estimator_

print('Coefficients:', best_lasso_model.coef_)
print('Intercept:', best_lasso_model.intercept_)

#stat sig: LASSO
secalc(X_scaledOLS, Ytr, best_lasso_model)
    
#create predicted values
predicted_values3 = best_lasso_model.predict(X_test_ols_all)

"""
4. LASSO model with only crypto quotes
"""

lasso_model2 = Lasso()

grid_search2 = GridSearchCV(lasso_model2, param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search2.fit(Xsc_crypto, Ycrypto)

best_alpha2 = grid_search2.best_params_['alpha']
# print best value of \alpha
print('Best alpha:', best_alpha2)

best_lasso_model2 = grid_search2.best_estimator_

print('Coefficients:', best_lasso_model2.coef_)
print('Intercept:', best_lasso_model2.intercept_)

#stat sig: LASSO
secalc(Xsc_crypto, Ycrypto, best_lasso_model2)
    
#create predicted values
predicted_values4 = best_lasso_model2.predict(X_test_ols_crypto)

"""
5. Data visualization of first two OLS regressions 
"""

#visualization of OLS with all quotes 

import matplotlib.pyplot as plt

plt.scatter(X_scaledOLS[:, 0], Ytr, label='Scaled Data')
xvals = np.array([-3, 3])
yvals = np.array([model.intercept_ - 3*model.coef_[0], model.intercept_ + 3*model.coef_[0]]) 

plt.plot(xvals, yvals, color = 'red', label = 'OLS Regression Line')

plt.title('BTC Price vs. Scaled Musk-Tweet Sentiment')
plt.xlabel('Musk-Tweet Sentiment, Scaled')
plt.ylabel('BTC Closing Price')
plt.legend()
plt.show()
plt.savefig('all_tweet_plot.png')

#visualization of OLS with only crypto quotes 

plt.scatter(Xsc_crypto [:, 0], Ycrypto, label='Scaled Data')
xvals = np.array([-3, 3])
yvals = np.array([cols_model.intercept_ - 3*cols_model.coef_[0], cols_model.intercept_ + 3*cols_model.coef_[0]]) 

plt.plot(xvals, yvals, color = 'red', label = 'OLS Regression Line')

plt.title('BTC Price vs. Scaled Musk-Tweet Sentiment (Crypto Tweets Only)')
plt.xlabel('Musk-Tweet Sentiment, Scaled')
plt.ylabel('BTC Closing Price')
plt.legend()
plt.show()
plt.savefig('crypto_tweet_plot.png')


#visualization of musk crypto tweet sentiment over time 

from datetime import datetime

# Sample data with dates
dates = [datetime.strptime(date, "%Y-%m-%d") for date in crypto_data.values[:, 0]]

# Plot the line
plt.plot(dates, crypto_data.values[:, 9], linestyle='-', color='b')

# Add title and labels
plt.title('Musk Tweet Sentiment Over Time (Crypto Tweets Only)')
plt.xlabel('Date')
plt.ylabel('Values')

# Show the plot
plt.show()



