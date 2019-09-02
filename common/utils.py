import numpy as np
import pandas as pd
import os
from collections import UserDict

from pandas.core.indexes.datetimes import DatetimeIndex
import datetime as dt

from sklearn.preprocessing.data import MinMaxScaler

from common.TimeseriesTensor import TimeSeriesTensor
from sklearn.utils import check_array


def load_data(data_dir):
    """Load the GEFCom 2014 energy load data"""

    target = pd.read_csv(os.path.join(data_dir, 'hourly-norm-data-2011.csv'), header=0, parse_dates={"timestamp": [0]})
    target = target.drop('timestamp', axis=1)

    # imf0 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-0.csv'), header=None)
    imf1 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-1.csv'), header=None)
    imf2 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-2.csv'), header=None)
    # imf3 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-3.csv'), header=None)
    # imf4 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-4.csv'), header=None)
    # imf5 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-5.csv'), header=None)
    # imf6 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-6.csv'), header=None)
    # imf7 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-7.csv'), header=None)
    # imf8 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-8.csv'), header=None)
    # imf9 = pd.read_csv(os.path.join(data_dir, 'eimf/norm-2011-eIMF-9.csv'), header=None)

    # Reindex the dataframe such that the dataframe has a record for every time point
    # between the minimum and maximum timestamp in the time series. This helps to
    # identify missing time periods in the data (there are none in this dataset).

    # energy.index = energy['timestamp']
    # energy = energy.reindex(pd.date_range(min(energy['timestamp']),
    #                                       max(energy['timestamp']),
    #                                       freq='H'))
    # energy = energy.drop('timestamp', axis=1)
    # df = pd.concat([target, imf0, imf1, imf2, imf3, imf4, imf5, imf6, imf7, imf8, imf9], axis=1)
    df = pd.concat([target, imf1, imf2], axis=1)
    # df = pd.concat([target, imf1, imf2, imf8, imf9], axis=1)
    # df.columns = ["load", "imf0", "imf1", "imf2", "imf3", "imf4", "imf5", "imf6", "imf7", "imf8", "imf9"]
    df.columns = ["load", "imf1", "imf2"]

    dt_idx = DatetimeIndex(freq='H', start='2011-01-01 00:00:00', end='2011-12-31 23:00:00')

    df.index = dt_idx

    return df


def load_data_full(data_dir, datasource='electricity', imfs_count=13, freq='H'):
    """Load the GEFCom 2014 energy load data"""
    target =None
    start_date = None
    end_date = None

    if datasource == 'electricity':
        target = pd.read_csv(os.path.join(data_dir, 'hourly_clean_electricity.csv'), header=0, parse_dates={"timestamp": [0]})
        start_date = min(target['timestamp'])
        end_date = max(target['timestamp'])
        target = target.drop('timestamp', axis=1)
    elif datasource == 'temperature':
        target = pd.read_csv(os.path.join(data_dir, 'temperature.csv'), header=0, parse_dates={"timestamp": [0]})
        start_date = min(target['timestamp'])
        end_date = max(target['timestamp'])
        target = target.drop('timestamp', axis=1)
    elif datasource == 'exchange-rate':
        target = pd.read_csv(os.path.join(data_dir, 'time_exchange_rate.csv'), header=0, parse_dates={"timestamp": [0]})
        start_date = min(target['timestamp'])
        end_date = max(target['timestamp'])
        target = target.drop('timestamp', axis=1)
    else:
        raise Exception('Not support the data source:', datasource)
    imfs =[]
    imf_lables = []
    imfs_dir = data_dir + '/' + datasource
    for i in range(imfs_count):
        file_path = imfs_dir + '/imfs/IMF-' + str(i) + '.csv'
        imf_i = pd.read_csv(file_path, header=None, dtype=np.float64)
        imfs.append(imf_i)
        imf_lables.append("imf" + str(i))

    df = pd.concat([target] + imfs, axis=1)

    df.columns = ["load"] + imf_lables

    # dt_idx = DatetimeIndex(freq='H', start='2011-01-01 00:00:00', end='2011-12-31 23:00:00')
    df.index = pd.date_range(start_date, end_date, freq=freq)

    return df




def load_data_one_source(data_dir):
    """Load the GEFCom 2014 energy load data"""

    target = pd.read_csv(os.path.join(data_dir, 'hourly-norm-data-2011.csv'), header=0, parse_dates={"timestamp": [0]})
    target = target.drop('timestamp', axis=1)


    dt_idx = DatetimeIndex(freq='H', start='2011-01-01 00:00:00', end='2011-12-31 23:00:00')
    target.columns = ["load"]

    target.index = dt_idx

    return target


def split_train_validation_test(multi_time_series_df, valid_start_time, test_start_time, features,
                                time_step_lag=1, horizon=1, target='target', time_format='%Y-%m-%d %H:%M:%S', freq='H'):

    if not isinstance(features, list) or len(features) < 1:
        raise Exception("Bad input for features. It must be an array of dataframe colummns used")

    train = multi_time_series_df.copy()[multi_time_series_df.index < valid_start_time]
    train = train[features]

    X_scaler = MinMaxScaler()

    if 'load' in features:
        y_scaler = MinMaxScaler()
        y_scaler.fit(train[['load']])
    else:
        y_scaler = MinMaxScaler()

        tg = train[target]
        y_scaler.fit(tg.values.reshape(-1, 1))

    train[features] = X_scaler.fit_transform(train)

    tensor_structure = {'X': (range(-time_step_lag + 1, 1), features)}
    train_inputs = TimeSeriesTensor(train, target=target, H=horizon, freq=freq, tensor_structure=tensor_structure)

    print(train_inputs.dataframe.head())


    look_back_dt = dt.datetime.strptime(valid_start_time, time_format) - dt.timedelta(hours=time_step_lag - 1)
    valid = multi_time_series_df.copy()[(multi_time_series_df.index >= look_back_dt) & (multi_time_series_df.index < test_start_time)]
    valid = valid[features]
    valid[features] = X_scaler.transform(valid)
    tensor_structure = {'X': (range(-time_step_lag + 1, 1), features)}
    valid_inputs = TimeSeriesTensor(valid, target=target, H=horizon, freq=freq, tensor_structure=tensor_structure)

    print(valid_inputs.dataframe.head())

    # test set
    # look_back_dt = dt.datetime.strptime(test_start_time, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=time_step_lag - 1)
    test = multi_time_series_df.copy()[test_start_time:]
    test = test[features]
    test[features] = X_scaler.transform(test)
    test_inputs = TimeSeriesTensor(test, target=target, H=horizon, freq=freq, tensor_structure=tensor_structure)

    print("time lag:", time_step_lag, "original_feature:", len(features))

    return train_inputs, valid_inputs, test_inputs, y_scaler


def mape(predictions, actuals):
    predictions = check_array(predictions)
    actuals = check_array(actuals)

    """Mean absolute percentage error"""
    return ( np.mean(np.abs(predictions - actuals) / actuals))


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df