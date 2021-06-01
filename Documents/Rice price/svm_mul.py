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
import seaborn as sns
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

def split_data(inf, m=1):

    X, y = split_sequence(inf, m)
    #
    X = np.array([entry[0] for entry in X]).reshape(-1,1)
    y = np.array([entry[0] for entry in y]).reshape(-1,1)
    #
    X_train = X[:math.ceil(len(X)*1)]
    y_train = y[:math.ceil(len(y)*1)]
    X_test = X[math.ceil(len(X)*1):]
    y_test = y[math.ceil(len(X)*1):]    
    
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
    
    df_test = df[len(df)-3:]
#    df = df[0:len(df)-3]
    
        ##correlation
#    df_new = df[['tot_r_stock', 'h_r_stock', 'c_r_stock', 'nfa', 'price']]
#    correlation = df_new.corr()
#
#    sns.heatmap(correlation, 
#         xticklabels=correlation.columns, 
#         yticklabels=correlation.columns,
#         cmap="Blues")


    ######### features
    inf = df.price.to_numpy()
    dates = df.dates.to_numpy()
    tot_r_stock = df.tot_r_stock.to_numpy()
#    h_r_stock = df.h_r_stock.to_numpy()
    c_r_stock = df.c_r_stock.to_numpy()
#    nfa = df.nfa.to_numpy()
    
    ########## Normalize
    inf, scaler1 = normalize_(inf)
    tot_r_stock, scaler = normalize_(tot_r_stock)
    c_r_stock, scaler = normalize_(c_r_stock)

    ##########
    X_train, y_train, X_test, y_test = split_data(inf)
    X_train_tot, y_train_tot, X_test_tot, y_test_tot = split_data(tot_r_stock)
    X_train_c, y_train_c, X_test_c, y_test_c = split_data(c_r_stock)
    
        ########## combine array to single input array
    input_x = np.concatenate((X_train, X_train_tot, X_train_c), axis=1)
    test_x = input_x[len(input_x)-3:]
    input_x = input_x[0:len(input_x)-3]
    y_test = y_train[len(y_train)-3:]
    y_train = y_train[0:len(y_train)-3]
    
    ########## Model
    clf = svm_model(input_x, y_train)
    
    m = 3
    predicted = []
    
    to_pred = np.array([input_x[-1]]).reshape(1,3)
    
    for i in np.arange(1, m+1, 1):
        pred = clf.predict(to_pred)
        
        predicted.append(pred[0])
        new_input = test_x[i-1]
        new_input[0] = pred[0]
        to_pred = new_input.reshape(1,3)
    
    predicted = np.array(predicted)
    predicted = scaler1.inverse_transform(np.array(predicted.reshape(-1,1)))
    
    print('Predicted = {}'.format(predicted))
    
        ########## RMSE
    rmse = np.sqrt(np.sum(np.power(df_test.price.values - predicted.reshape(3,), 2))/float(len(df_test.price.values)))
    print('#####', rmse, '#####')

    ########## MAE
    mae = MAE(df_test.price.values, predicted.reshape(3))
    print('##### MAE: ', mae, '#####')
          
