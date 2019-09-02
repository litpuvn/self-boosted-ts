#!/usr/bin/env python
# coding: utf-8
from keras.callbacks import EarlyStopping
import pandas as pd
from keras.layers.core import RepeatVector
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from pandas import DatetimeIndex

from common.TimeseriesTensor import TimeSeriesTensor
from common.gp_log import store_training_loss, store_predict_points, flatten_test_predict
from common.utils import load_data, split_train_validation_test, mape, load_data_one_source




from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

def RMSE(x):
    return sqrt(x)

if __name__ == '__main__':
    time_step_lag = 12
    HORIZON = 1



    data = pd.read_csv('/home/ope/Documents/Projects/self-boosted-ts/data/solar-power/Actual_32.05_-94.15_2006_UPV_95MW_5_Min.csv', parse_dates=['time'])
    data.index = data['time']

    data = data.drop('time', axis=1)
    data = data[['power']]

    data.to_csv("solar-power.csv")

    print(data.head())