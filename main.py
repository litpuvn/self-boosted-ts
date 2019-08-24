from common.TimeseriesTensor import TimeSeriesTensor
from common.utils import load_data, split_train_validation_test

time_step_lag = 6
HORIZON = 3

data_dir = 'data/'
multi_time_series = load_data(data_dir)
print(multi_time_series.head())


valid_start_dt = '2011-09-01 00:00:00'
test_start_dt = '2011-11-01 00:00:00'

train, valid, test = split_train_validation_test(multi_time_series,
                                                 valid_start_time=valid_start_dt,
                                                 test_start_time=test_start_dt,
                                                 time_step_lag=time_step_lag,
                                                 features=["imf1", "imf2"]
                                                 )

# train = energy.copy()[energy.index < valid_start_dt][['load']]

from sklearn.preprocessing import MinMaxScaler
# transforming data
# scaler = MinMaxScaler()
# scaler.fit(train[['load']])
# train[['load']] = scaler.transform(train)
#
# tensor_structure = {'X':(range(-time_step_lag+1, 1), ['load'])}
# train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)