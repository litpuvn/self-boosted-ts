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
from common.utils import load_data, split_train_validation_test, mape, load_data_one_source, load_data_full, \
    load_dwt_seasonal_data

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import os
import numpy as np

def RMSE(x):
    return sqrt(x)

if __name__ == '__main__':


    time_step_lag = 12

    EPOCHS = 50
    HORIZON = 1
    datasource = 'electricity'
    # datasource = 'temperature'
    # datasource = 'exchange-rate'
    mode = 'additive'
    # mode = 'multiplicative'
    # predict_component = 'Residual'
    # predict_component = 'Trend'
    # predict_component = 'Seasonal'
    predict_component = 'Observed'

    data_dir = '/home/long/TTU-SOURCES/self-boosted-ts/data/seasonal'
    output_dir = '/home/long/TTU-SOURCES/self-boosted-ts/output/electricity/dwt/seq2seq'


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/model_checkpoint', exist_ok=True)

    multi_time_series, valid_start_dt, test_start_dt, freq = load_dwt_seasonal_data(data_dir, datasource=datasource,
                                                                                    mode=mode,
                                                                                    target_component=predict_component)
    print(multi_time_series.head())

    features = ["load"]
    targets = ["load"]

    time_format='%Y-%m-%d %H:%M:%S'
    if freq == 'd':
        time_format = '%Y-%m-%d'

    train_inputs, valid_inputs, test_inputs, y_scaler = split_train_validation_test(multi_time_series,
                                                                                    valid_start_time=valid_start_dt,
                                                                                    test_start_time=test_start_dt,
                                                                                    time_step_lag=time_step_lag,
                                                                                    horizon=HORIZON,
                                                                                    features=features,
                                                                                    target=targets,
                                                                                    freq=freq,
                                                                                    time_format=time_format
                                                                                    )

    X_train = train_inputs['X']
    y_train = train_inputs['target_load']

    X_valid = valid_inputs['X']
    y_valid = valid_inputs['target_load']

    print("train_X shape", X_train.shape)
    print("valid_X shape", X_valid.shape)

    from keras.models import Model, Sequential
    from keras.layers import Conv1D, Dense, Flatten
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    LATENT_DIM = 5
    KERNEL_SIZE = 2

    BATCH_SIZE = 32

    model = Sequential()
    model.add(GRU(LATENT_DIM, input_shape=(time_step_lag, 1)))
    model.add(RepeatVector(1))
    model.add(GRU(LATENT_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', patience=5)


    # Test the model
    X_test = test_inputs['X']
    y1_test = test_inputs['target_load']
    y1_test = y_scaler.inverse_transform(y1_test)

    if not os.path.exists(output_dir + '/original_' + predict_component + '_lag' + str(time_step_lag) + '.csv'):
        np.savetxt(output_dir + '/original_' + predict_component + '_lag' + str(time_step_lag) + '.csv', y1_test,
                   delimiter=',', fmt='%s')

    if predict_component == 'Observed':
        exit(0)

    history = model.fit(X_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_valid, y_valid),
                        callbacks=[earlystop],
                        verbose=1)

    # Test the model
    X_test = test_inputs['X']
    y1_preds = model.predict(X_test)
    y1_preds = y_scaler.inverse_transform(y1_preds)

    y1_test, y1_preds = flatten_test_predict(y1_test, y1_preds)

    np.savetxt(output_dir + '/predicted_' + predict_component + '_lag' + str(time_step_lag) + '_h' + str(HORIZON) + '.csv', y1_preds, delimiter=',', fmt='%s')

    mse = mean_squared_error(y1_test, y1_preds)

    rmse_predict = RMSE(mse)
    evs = explained_variance_score(y1_test, y1_preds)
    mae = mean_absolute_error(y1_test, y1_preds)
    # msle = mean_squared_log_error(y1_test, y1_preds)
    meae = median_absolute_error(y1_test, y1_preds)
    r_square = r2_score(y1_test, y1_preds)

    mape_v = mape(y1_preds.reshape(-1, 1), y1_test.reshape(-1, 1))

    # print("mse:", mse, 'rmse_predict:', rmse_predict, "mae:", mae, "mape:", mape_v, "r2:", r_square, "msle:", msle, "meae:", meae, "evs:", evs)
    print('rmse_predict:', rmse_predict, "evs:", evs, "mae:", mae,
          "mse:", mse, "meae:", meae, "r2:", r_square, "mape", mape_v)