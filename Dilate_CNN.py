#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

data = pd.read_csv('data-2011.csv', parse_dates=['time'])
print (data.head())
print ('\n Data Types:')
print (data.dtypes)


data = pd.read_csv('data-2011.csv', parse_dates=['time'])
data.index = data['time']
data = data.reindex(pd.date_range(min(data['time']), max(data['time']), freq='H'))
data = data.drop('time', axis=1)

data = data[['avg_electricity']]
data.head()

data.dtypes

#validation and test start dates
valid_start_dt = '2011-10-01 0:00:00'
test_start_dt = '2011-11-01 0:00:00'

#separate and plot the three data
data[data.index < valid_start_dt][['avg_electricity']].rename(columns={'avg_electricity':'train'})     .join(data[(data.index >=valid_start_dt) & (data.index < test_start_dt)][['avg_electricity']]           .rename(columns={'avg_electricity':'validation'}), how='outer')     .join(data[test_start_dt:][['avg_electricity']].rename(columns={'avg_electricity':'test'}), how='outer')     .plot(y=['train', 'validation', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('time', fontsize=12)
plt.ylabel('avg_electricity', fontsize=12)
plt.show()

