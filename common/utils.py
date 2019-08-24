import numpy as np
import pandas as pd
import os
from collections import UserDict

from pandas.core.indexes.datetimes import DatetimeIndex
import datetime as dt

from sklearn.preprocessing.data import MinMaxScaler

from common.TimeseriesTensor import TimeSeriesTensor


def load_data(data_dir):
    """Load the GEFCom 2014 energy load data"""

    target = pd.read_csv(os.path.join(data_dir, 'hourly-norm-data-2011.csv'), header=0, parse_dates={"timestamp": [0]})
    target = target.drop('timestamp', axis=1)

    imf1 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-1.csv'), header=None)

    imf2 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-2.csv'), header=None)

    # Reindex the dataframe such that the dataframe has a record for every time point
    # between the minimum and maximum timestamp in the time series. This helps to
    # identify missing time periods in the data (there are none in this dataset).

    # energy.index = energy['timestamp']
    # energy = energy.reindex(pd.date_range(min(energy['timestamp']),
    #                                       max(energy['timestamp']),
    #                                       freq='H'))
    # energy = energy.drop('timestamp', axis=1)
    df = pd.concat([target, imf1, imf2], axis=1)
    df.columns = ["load", "imf1", "imf2"]

    dt_idx = DatetimeIndex(freq='H', start='2011-01-01 00:00:00', end='2011-12-31 23:00:00')

    df.index = dt_idx

    return df


def split_train_validation_test(multi_time_series_df, valid_start_time, test_start_time, features,
                                time_step_lag=1, HORIZON=1):

    if not isinstance(features, list) or len(features) < 1:
        raise Exception("Bad input for features. It must be an array of dataframe colummns used")

    train = multi_time_series_df.copy()[multi_time_series_df.index < valid_start_time]
    X_scaler = MinMaxScaler()
    train[['load', 'imf1', 'imf2']] = X_scaler.fit_transform(train)

    tensor_structure = {'X': (range(-time_step_lag + 1, 1), ['load', 'imf1', 'imf2'])}
    train_inputs = TimeSeriesTensor(train, target='load', H=HORIZON, tensor_structure=tensor_structure)

    print(train_inputs.dataframe.head())


    look_back_dt = dt.datetime.strptime(valid_start_time, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=time_step_lag - 1)
    valid = multi_time_series_df.copy()[(multi_time_series_df.index >= look_back_dt) & (multi_time_series_df.index < test_start_time)]
    valid[['load', 'imf1', 'imf2']] = X_scaler.transform(valid)
    tensor_structure = {'X': (range(-time_step_lag + 1, 1), ['load', 'imf1', 'imf2'])}
    valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

    print(valid_inputs.dataframe.head())

    # test set
    # look_back_dt = dt.datetime.strptime(test_start_time, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=time_step_lag - 1)
    test = multi_time_series_df.copy()[test_start_time:]
    test[['load', 'imf1', 'imf2']] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)

    print("time lag:", time_step_lag, "original_feature:", len(features))

    return train_inputs, valid_inputs, test_inputs


def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df