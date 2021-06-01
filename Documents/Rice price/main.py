#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:32:35 2021

@author: marvin-corp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import kpss
import seaborn as sns
import matplotlib.dates as md
from sklearn import preprocessing




tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

colors_ = [(22,82,109),(153,27,30),(248,153,29)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

for i in range(len(colors_)):
    r_d,g_d,b_d = colors_[i]
    colors_[i] = (r_d / 255., g_d / 255., b_d / 255.)
    
    
def month_iterator(start_date, end_date):
    d = np.array(pd.date_range(start_date, end_date, freq = 'M'))
    
    return d

def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff



df = pd.read_csv('df.csv')
dates = month_iterator('1995-01-01', '2021-01-01')
df['dates'] = dates

x_rate = pd.read_csv('exchange_rate.csv')
x_rate['dollar'] = pd.to_numeric(x_rate['dollar'], errors='coerce')
x_rate['Period']= pd.to_datetime(x_rate['Period'])

x_rate = x_rate.set_index('Period')
x_rate=x_rate.resample('M', how='mean')
x_rate.reset_index(level=0, inplace=True)

x_rate.rename(columns={'Period':'dates'}, inplace=True)
final = pd.merge(df, x_rate, on='dates')
final['imports_doltopeso'] = final['imports'] * final['dollar']

df = final
#df['production'] = df.production / 1000000  

min_max_scaler = preprocessing.MinMaxScaler()
df['scaled_imports_doltopeso'] = min_max_scaler.fit_transform(df['imports_doltopeso'].values.reshape(len(df),1)) * 50

min_max_scaler = preprocessing.MinMaxScaler()
df['scaled_imports'] = min_max_scaler.fit_transform(df['imports'].values.reshape(len(df),1)) * 50

min_max_scaler = preprocessing.MinMaxScaler()
df['scaled_nfa'] = min_max_scaler.fit_transform(df['nfa'].values.reshape(len(df),1)) * 50

production = df.production.values.astype(np.double)
production_ = np.isfinite(production)

fig, ax = plt.subplots()


#
##ax.plot_date(df.dates.values[production_], production[production_], linestyle='-', marker='o',color='r',)
#
ax.plot(df.dates, df.price, label = 'Inflation', color = colors_[1], linewidth=2)
plt.axhline(y=0, color = 'black', linestyle='--', label = '0-inflation')
plt.axvline(x = '2008-07-29', color = 'dimgrey')
plt.axvline(x = '2014-07-21', color = 'dimgrey')
plt.axvline(x = '2018-10-29', color = 'dimgrey')
plt.axvline(x = '2019-10-21', color = 'dimgrey')
plt.ylabel('Inflation rate',color=colors_[1], fontsize = 18)

##plt.legend()
ax2 = ax.twinx()
#ax2.plot(df.dates, df.price)
ax2.plot(df.dates, df.scaled_nfa, color = colors_[0], label = 'NFA stock', alpha=0.8)
#ax2.plot(df.dates, df.scaled_imports_doltopeso, color = 'orange',marker = 'd',alpha = 0.6, label = 'Import value (Dollar to Peso - scaled)')
ax2.plot(df.dates, df.dollar, color = 'green',alpha = 0.6, label = 'Dollar to peso exchange rate')
ax2.plot(df.dates, df.scaled_imports, color = colors_[2], label = 'Imports value (rescaled)', alpha=0.8)
plt.legend(fontsize = 15,loc='upper left')

##   ax2.plot(df.dates, df_new.c_r_stock, color = 'yellow')
#
##
ax.xaxis.set_major_formatter(md.DateFormatter("%d%b'%y"))
ax.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='y', which='major', labelsize=15)
###
####fig,ax = plt.subplots()
####
####ax.plot(x_rate.Period, x_rate.US)
####ax2 = ax.twinx()
####ax2.plot(df.dates, df.price)