#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 08:58:49 2021

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
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")

def getfeatures(norm, m, train_ratio):
    
    x = []
    y = []
    
    for i in range(len(norm) - m):
        x.append(norm[i:i+m])
        y.append(norm[i+m])
    
    x = np.array(x)
    y = np.array(y)


    #### Get last index of training set

    last_index = int(len(x)*train_ratio)

    #### Split into training set and test set

    train_x = x[0:last_index]

    train_y = y[0:last_index]

    test_x = x[last_index:]

    test_y = y[last_index:]

    

    return train_x, train_y, test_x, test_y


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


def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d

def normalize_(array):
   
    scaler = MinMaxScaler(feature_range = (0,1))
    inf = scaler.fit_transform(array.reshape(-1,1))
    
    return inf, scaler

def split_data(inf, m=2):

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

def GetOptimalCLF(train_x, train_y, rand_starts = 8):
    
    min_loss = 1e10
    
    for i in range(rand_starts):
        
        n_input = train_x.shape[1]
        
        print("Iteration number {}".format(i+1))
        
        clf = MLPRegressor(hidden_layer_sizes = (int(round(20*np.sqrt(n_input),0)),5), 
                           activation = 'relu', solver = 'adam', learning_rate = 'adaptive', 
                           max_iter = 1000000000000, tol =  1e-10, early_stopping = True,
                           validation_fraction = 1/3)
        
        clf.fit(train_x,train_y)
        
        cur_loss = clf.loss_
        
        if cur_loss < min_loss:
            
            min_loss = cur_loss
            max_clf = clf
        
        print("Current loss {}".format(min_loss))
        
    return max_clf


df = pd.read_csv('df.csv')

dates = month_iterator('1995-01-01', '2021-01-01')    
df['dates'] = dates
########## features
inf = df.price.as_matrix()
dates = df.dates.as_matrix()

########## Convert to 1d vector
inf = np.reshape(inf, (len(inf),1))
dates = np.reshape(dates, (len(dates),1))

########## Normalize
inf, scaler = normalize_(inf)


X_train, y_train, X_test, y_test = split_data(inf)


clf = GetOptimalCLF(X_train, y_train)

########## Predict
#pred_train = clf.predict(train_x)
pred_test = clf.predict(X_test)

########## revert back
pred_test = scaler.inverse_transform(np.array(pred_test.reshape(-1,1)))
y_test = scaler.inverse_transform(np.array(y_test.reshape(-1,1)))


df_final = pd.DataFrame({'pred':pred_test.reshape(len(pred_test),),
                         'act':y_test.reshape(len(y_test),)})
    
    
    
df_final['act_bin'] = np.where((df_final['act'] < df_final.act.shift(-1)),1,0)
df_final['pred_bin'] = np.where((df_final['act'] < df_final.pred.shift(-1)),1,0)


accuracy = accuracy_score(df_final.act_bin, df_final.pred_bin)
recall = recall_score(df_final.act_bin, df_final.pred_bin)
precision = precision_score(df_final.act_bin, df_final.pred_bin)
f1 = f1_score(df_final.act_bin, df_final.pred_bin)