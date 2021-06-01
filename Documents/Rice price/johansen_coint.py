#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:13:45 2021

@author: marvin-corp
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank
from sklearn.metrics import mean_absolute_error as MAE

def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d


df = pd.read_csv('df.csv')

dates = month_iterator('1995-01-01', '2021-01-01')


df['dates'] = dates

df = df[['tot_r_stock', 'h_r_stock', 'c_r_stock', 'nfa', 'price']]
df_test = df[-3:]
df = df[0:309]
#
#df = df[['price','tot_r_stock']]

lag_order = select_order(df, 5,deterministic="ci")

coint = select_coint_rank(df, 0, 3)
print(coint)

#model = VECM(df, deterministic="ci", diff_lags=3, coint_rank=4)  # =1

vecm = VECM(endog = df, k_ar_diff = 3, coint_rank = 4, deterministic = 'ci')
vecm_fit = vecm.fit()
#vecm_fit.predict(steps=10)
vecm_fit.plot_forecast(steps=3, n_last_obs=12)

forecast, lower, upper = vecm_fit.predict(steps=3, alpha=0.05)
#for text, vaĺues in zip(("forecast", "lower", "upper"), vecm_fit.predict(steps=5, alpha=0.05)):
#    print(text+":", vaĺues, sep="\n")
#    
#

#num_periods = 30
#ir = vecm_fit.irf(periods=num_periods)
#ir.plot(plot_stderr=False)
predicted = []
for i in range(len(forecast)):
    predicted.append(forecast[i][-1])

predicted = np.array(predicted)


########## RMSE
rmse = np.sqrt(np.sum(np.power(df_test.price.values - predicted, 2))/float(len(df_test.price.values)))
print('##### RMSE: ', rmse, '#####')
      
########## MAE
mae = MAE(df_test.price.values, predicted)
print('##### MAE: ', mae, '#####')

