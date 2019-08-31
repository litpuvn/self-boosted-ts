from keras.callbacks import EarlyStopping
import pandas as pd
from pandas import DatetimeIndex

from common.TimeseriesTensor import TimeSeriesTensor
from common.gp_log import store_training_loss, store_predict_points, flatten_test_predict
from common.utils import load_data, split_train_validation_test, mape, load_data_one_source
#from kgp.metrics import root_mean_squared_error as RMSE
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
    HORIZON = 1

    target = pd.read_csv('/home/ope/Documents/Projects/self-boosted-ts/data/exchange_rate.txt', header=0, usecols=[0])

    dt_idx = DatetimeIndex(freq='d', start='1990-01-01 00:00:00', periods=7588)


    target.index = dt_idx

    target.to_csv("time_exchange_rage.csv")
    training_end_index = '2002-06-17 00:00:00'
    print(target.head())

    multi_time_series = target

    print("count data rows=", multi_time_series.count)

    valid_start_dt = '2002-06-18'
    test_start_dt = '2006-08-13'

    train_inputs, valid_inputs, test_inputs, y_scaler = split_train_validation_test(multi_time_series, valid_start_time=valid_start_dt, test_start_time=test_start_dt,
                                                                                    time_step_lag=time_step_lag, horizon=HORIZON,
                                                                                    features=["rate"], target='rate',
                                                                                    time_format='%Y-%m-%d',
                                                                                    freq='d'

                                                        )
    X_train = train_inputs['X']
    y_train = train_inputs['target_rate']

    X_valid = valid_inputs['X']
    y_valid = valid_inputs['target_rate']


    print("train_X shape", X_train.shape)
    print("valid_X shape", X_valid.shape)

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
    y1_test = test_inputs['target_rate']

    y1_preds = model.predict(X_test)

    y1_test = y_scaler.inverse_transform(y1_test)
    y1_preds = y_scaler.inverse_transform(y1_preds)

    y1_test, y1_preds = flatten_test_predict(y1_test, y1_preds)
    mse = mean_squared_error(y1_test, y1_preds)

    rmse_predict = RMSE(mse)
    evs = explained_variance_score(y1_test, y1_preds)
    mae = mean_absolute_error(y1_test, y1_preds)
    msle = mean_squared_log_error(y1_test, y1_preds)
    meae = median_absolute_error(y1_test, y1_preds)
    r_square = r2_score(y1_test, y1_preds)

    print('rmse_predict:', rmse_predict, "evs:", evs, "mae:", mae,
          "mse:", mse, "msle:", msle, "meae:", meae, "r2:", r_square)
