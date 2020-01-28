from keras.callbacks import EarlyStopping
import pandas as pd
from keras.layers.core import RepeatVector
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from pandas import DatetimeIndex

from common.TimeseriesTensor import TimeSeriesTensor
from common.gp_log import store_training_loss, store_predict_points, flatten_test_predict
from common.utils import load_data, split_train_validation_test, mape, load_data_one_source, load_data_full

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
    time_step_lag = 6
    HORIZON = 5

    imfs_count = 0 # set equal to zero for not considering IMFs features

    data_dir = '/home/long/TTU-SOURCES/self-boosted-ts/data'
    output_dir = '/home/long/TTU-SOURCES/self-boosted-ts/output/exchange-rate'

    multi_time_series = load_data_full(data_dir, datasource='exchange-rate', imfs_count=imfs_count, freq='d')
    print(multi_time_series.head())


    print("count data rows=", multi_time_series.count)

    valid_start_dt = '2002-06-18'
    test_start_dt = '2006-08-13'

    train_inputs, valid_inputs, test_inputs, y_scaler = split_train_validation_test(multi_time_series,
                                                                                    valid_start_time=valid_start_dt,
                                                                                    test_start_time=test_start_dt,
                                                                                    time_step_lag=time_step_lag,
                                                                                    horizon=HORIZON,
                                                                                    features=["load"],
                                                                                    target=['load'],
                                                                                    time_format='%Y-%m-%d',
                                                                                    freq='d')

    X_train = train_inputs['X']
    y_train = train_inputs['target_load']

    X_valid = valid_inputs['X']
    y_valid = valid_inputs['target_load']


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
    BATCH_SIZE = 32
    EPOCHS = 100

    model = Sequential()
    model.add(GRU(LATENT_DIM, input_shape=(time_step_lag, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train,
                        y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_valid, y_valid),
              callbacks=[earlystop],
              verbose=1)

    # Test the model
    X_test = test_inputs['X']
    y1_test = test_inputs['target_load']

    y1_preds = model.predict(X_test)

    y1_test = y_scaler.inverse_transform(y1_test)
    y1_preds = y_scaler.inverse_transform(y1_preds)

    y1_test, y1_preds = flatten_test_predict(y1_test, y1_preds)
    mse = mean_squared_error(y1_test, y1_preds)

    rmse_predict = RMSE(mse)
    evs = explained_variance_score(y1_test, y1_preds)
    mae = mean_absolute_error(y1_test, y1_preds)
    mse = mean_squared_error(y1_test, y1_preds)
    msle = mean_squared_log_error(y1_test, y1_preds)
    meae = median_absolute_error(y1_test, y1_preds)
    r_square = r2_score(y1_test, y1_preds)
    mape_v = mape(y1_preds.reshape(-1, 1), y1_test.reshape(-1, 1))

    # print("mse:", mse, 'rmse_predict:', rmse_predict, "mae:", mae, "mape:", mape_v, "r2:", r_square, "msle:", msle, "meae:", meae, "evs:", evs)

    # c_mape = mape(y1_test, y1_preds)
    # print("mape:", c_mape)
    print('rmse_predict:', rmse_predict, "evs:", evs, "mae:", mae,
          "mse:", mse, "meae:", meae, "r2:", r_square, "mape", mape_v)