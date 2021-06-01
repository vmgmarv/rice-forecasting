# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 08:37:07 2020

@author: GABRIELVC
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
import math
import matplotlib.dates as md
import warnings
warnings.filterwarnings("ignore")

print("Running script")

def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def read_data():
    
    df_inf = pd.read_csv('ncr_cpi.csv')
    
    df_inf = df_inf.dropna()
    df_inf.rename(columns={'Unnamed: 0': 'Year', 'Unnamed: 1': 'Month',
                       'Unnamed: 2': 'Month_number'}, inplace = True)
        
    df_inf['yoy_ALL'] = ((df_inf.ALL.shift(-12) - df_inf.ALL) / df_inf.ALL) * 100
    df_inf['yoy_ALL'] = df_inf.yoy_ALL.shift(12)
    
    df_inf['Shocks'] = np.where((df_inf['yoy_ALL'] - df_inf['yoy_ALL'].shift(1) > 0),1,0)
    
    dates = month_iterator('1994-01-01', '2020-11-01')
    
    df_inf['Dates'] = dates
    df_inf['MonthYear'] = df_inf.Dates.dt.strftime('%b %Y')
    df_inf = df_inf.dropna()
    
    df_inf['Dates2'] = df_inf['Dates'].map(md.date2num) ##### datetime to integer

    return df_inf

def normalize_(array):
   
    scaler = MinMaxScaler(feature_range = (0,1))
    inf = scaler.fit_transform(array.reshape(-1,1))
    
    return inf, scaler

def split_data(inf, m=3):

    X, y = split_sequence(inf, m)
    #
    X = np.array([entry[0] for entry in X]).reshape(-1,1)
    y = np.array([entry[0] for entry in y]).reshape(-1,1)
    #
    X_train = X[:math.ceil(len(X)*0.80)]
    y_train = y[:math.ceil(len(y)*0.80)]
    X_test = X[math.ceil(len(X)*0.80):]
    y_test = y[math.ceil(len(X)*0.80):]    
    
    return X_train, y_train, X_test, y_test


def svm_model(X_train, y_train):
    
    param_grid = {"C": np.linspace(10**(-2),10**3,100),
             'gamma': np.linspace(0.0001,1,20)}

    mod = SVR(epsilon = 0.000001,kernel='rbf')
    model = GridSearchCV(estimator = mod, param_grid = param_grid,
                         n_jobs=-1,scoring = "neg_mean_squared_error",verbose = 0)
    
    best_model = model.fit(X_train, y_train.ravel())
    
    return best_model

def abs_error(df_test):
    
    df_test['abs_error'] = abs(df_test.Actual.values - df_test.Predicted.values)
    
    return df_test.abs_error.values

if __name__ == '__main__':
    
    #############################################################
    df = pd.read_csv('df.csv')

    dates = month_iterator('1995-01-01', '2021-01-01')    
    df['dates'] = dates

    ######### features
    inf = df.price.to_numpy()
    dates = df.dates.to_numpy()
    
    inf_test = inf[len(df)-3:]
    inf = inf[0:len(df)-3]
#    ################################# temporary
#    inf = np.append(inf, 2.2)
#    dates = np.append(dates, np.datetime64('2020-07-31'))
#    #################################
#    
    
    ########## Convert to 1d vector
    inf = np.reshape(inf, (len(inf),1))
    dates = np.reshape(dates, (len(dates),1))
    
    ########## Normalize
    inf, scaler = normalize_(inf)
    
    ##########
    X_train, y_train, X_test, y_test = split_data(inf)
    X_d_tr, y_d_tr, X_d_ts, y_d_ts = split_data(dates) 
    
    ########## Model
    clf = svm_model(X_train, y_train)
    
    m = 3
    predicted = []
    
    to_pred = np.array([inf[-1]]).reshape(1,1)
    
    for i in np.arange(1, m+1, 1):
        pred = clf.predict(to_pred)
        
        predicted.append(pred[0])
        to_pred = pred.reshape(1,1)
    
    predicted = np.array(predicted)
    predicted = scaler.inverse_transform(np.array(predicted.reshape(-1,1)))
    
    print('Predicted = {}'.format(predicted))
    
        ########## RMSE
    rmse = np.sqrt(np.sum(np.power(inf_test - predicted.reshape(3,), 2))/float(len(inf_test)))
    print('#####', rmse, '#####')

    ########## MAE
    mae = MAE(inf_test, predicted.reshape(3))
    print('##### MAE: ', mae, '#####')
          
