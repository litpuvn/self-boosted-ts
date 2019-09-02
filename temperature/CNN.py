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



    data = pd.read_csv('/home/ope/Documents/Projects/self-boosted-ts/data/temperature.csv', parse_dates=['Date_Time'])
    data.index = data['Date_Time']
    data = data.reindex(pd.date_range(min(data['Date_Time']), max(data['Date_Time']), freq='H'))
    data = data.drop('Date_Time', axis=1)

    data = data[['temperature']]

    multi_time_series = data

    valid_start_dt = '2004-10-30 14:00:00'
    test_start_dt = '2005-01-16 13:00:00'

    train_inputs, valid_inputs, test_inputs, y_scaler = split_train_validation_test(multi_time_series,
                                                                                    valid_start_time=valid_start_dt,
                                                                                    test_start_time=test_start_dt,
                                                                                    time_step_lag=time_step_lag,
                                                                                    horizon=HORIZON,
                                                                                    features=["temperature"],
                                                                                    target='temperature'
                                                                                    )

    X_train = train_inputs['X']
    y_train = train_inputs['target_temperature']

    X_valid = valid_inputs['X']
    y_valid = valid_inputs['target_temperature']

    # input_x = train_inputs['X']
    print("train_X shape", X_train.shape)
    print("valid_X shape", X_valid.shape)
    # print("target shape", y_train.shape)
    # print("training size:", len(train_inputs['X']), 'validation', len(valid_inputs['X']), 'test size:', len(test_inputs['X']) )
    # print("sum sizes", len(train_inputs['X']) + len(valid_inputs['X']) + len(test_inputs['X']))

    ## build CNN
    from keras.models import Model, Sequential
    from keras.layers import Conv1D, Dense, Flatten
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    LATENT_DIM = 5
    KERNEL_SIZE = 2

    BATCH_SIZE = 32
    EPOCHS = 100

    model = Sequential()
    # conv = Conv1D(kernel_size=3, filters=5, activation='relu')(x)

    model.add(
        Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=1,
               input_shape=(time_step_lag, 1)))
    model.add(
        Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=2))
    model.add(
        Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=4))
    model.add(Flatten())
    model.add(Dense(HORIZON, activation='linear'))

    model.summary()

    model.compile(optimizer='Adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    earlystop = EarlyStopping(monitor='val_mse', patience=5)
    history = model.fit(X_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystop],
                        verbose=1)

    # Test the model
    X_test = test_inputs['X']
    y1_test = test_inputs['target_temperature']

    y1_preds = model.predict(X_test)

    y1_test = y_scaler.inverse_transform(y1_test)
    y1_preds = y_scaler.inverse_transform(y1_preds)

    y1_test, y1_preds = flatten_test_predict(y1_test, y1_preds)
    mse = mean_squared_error(y1_test, y1_preds)

    rmse_predict = RMSE(mse)
    evs = explained_variance_score(y1_test, y1_preds)
    mae = mean_absolute_error(y1_test, y1_preds)
    mse = mean_squared_error(y1_test, y1_preds)

    meae = median_absolute_error(y1_test, y1_preds)
    r_square = r2_score(y1_test, y1_preds)
    mape_v = mape(y1_preds.reshape(-1, 1), y1_test.reshape(-1, 1))

    print("mse:", mse, 'rmse_predict:', rmse_predict, "mae:", mae, "mape:", mape_v, "r2:", r_square,
          "meae:", meae, "evs:", evs)




