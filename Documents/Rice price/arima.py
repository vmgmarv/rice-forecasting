#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:05:55 2020

@author: marvin-corp
"""

import numpy as np
import pandas as pd
import matplotlib.dates as md
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
import datetime
from sklearn.metrics import mean_absolute_error as MAE

def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d


df_inf = pd.read_csv('df.csv')

    
dates = month_iterator('1994-01-01', '2020-01-30')

df_inf['Dates'] = dates
df_inf['MonthYear'] = df_inf.Dates.dt.strftime('%b %Y')
df_inf = df_inf.dropna()

df_inf['Dates2'] = df_inf['Dates'].map(md.date2num) ##### datetime to integer


df = df_inf[['Dates', 'price']].dropna()

df['Dates'] = df['Dates'].apply(lambda x: datetime.date(x.year,x.month,x.day))
df.index = df.Dates
df = df.drop(['Dates'], axis = 1)

df_train = df[0:len(df)-4]
df_test = df[len(df)-4:]
#
#result = seasonal_decompose(df.inf, model='additive')

predicted = []

for i in range(1,5):
    
    print('predicting: ' ,'{}/12'.format(i))
    stepwise_model = auto_arima(df_train, start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)

    stepwise_model.fit(df_train)

    future_forecast = stepwise_model.predict(n_periods=1)
    
    predicted.append(future_forecast[0])
    
    temp_df = pd.DataFrame({'Dates':[df_test.index[i-1]], 'price':[future_forecast[0]]})
    temp_df = temp_df.set_index('Dates')
    
    df_train = pd.concat([df_train, temp_df])
    


df_test['predicted'] = predicted

df_test['price'] = df_test.price.shift(1)
df_test = df_test.dropna()

########## RMSE
rmse = np.sqrt(np.sum(np.power(df_test.price.values - df_test.predicted.values, 2))/float(len(df_test.price.values)))
print('##### RMSE: ', rmse, '#####')
      
########## MAE
mae = MAE(df_test.price.values, df_test.predicted.values)
print('##### MAE: ', mae, '#####')

