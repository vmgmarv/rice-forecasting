# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:49:27 2020

@author: GABRIELVC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as md
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.dates as mdates
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
    X_train = X[:math.ceil(len(X)*1)]
    y_train = y[:math.ceil(len(y)*1)]
    X_test = X[math.ceil(len(X)*1):]
    y_test = y[math.ceil(len(X)*1):]    
    
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

if __name__ == "__main__":
    
    df = pd.read_csv('df.csv')

    dates = month_iterator('1995-01-01', '2021-01-01')    
    df['dates'] = dates

    ######### features
    inf = df.price.to_numpy()
    dates = df.dates.to_numpy()
    
    inf_test = inf[len(df)-3:]
    inf = inf[0:len(df)-3]
    ########## Convert to 1d vector
    inf = np.reshape(inf, (len(inf),1))
    dates = np.reshape(dates, (len(dates),1))
    
    ########## Normalize
    inf_n, scaler = normalize_(inf)

    ########## Split data
    X_train, y_train, X_test, y_test = split_data(inf_n)
    
    ########## Model
    clf = GetOptimalCLF(X_train, y_train)
    
    
    m = 3 ## number of data points to be predicted
    predicted = []
    to_pred = np.array([inf_n[-1]]).reshape(-1,1)

    for i in np.arange(1, m+1, 1):
        pred = clf.predict(to_pred)
        predicted.append(pred)
#        predicted.append(pred[-1])
        to_pred = pred.reshape(-1,1)
    
    predicted = np.array(predicted)
    predicted = scaler.inverse_transform(np.array(predicted.reshape(-1,1)))
    
    print('Predicted = {}'.format(predicted))

            ########## RMSE
    rmse = np.sqrt(np.sum(np.power(inf_test - predicted.reshape(3,), 2))/float(len(inf_test)))
    print('#####', rmse, '#####')

    ########## MAE
    mae = MAE(inf_test, predicted.reshape(3))
    print('##### MAE: ', mae, '#####')


