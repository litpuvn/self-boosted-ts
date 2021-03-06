from keras.callbacks import EarlyStopping
import pandas as pd
from common.TimeseriesTensor import TimeSeriesTensor
from common.gp_log import store_training_loss, store_predict_points, flatten_test_predict
from common.utils import load_data, split_train_validation_test, load_data_full, mape
from ts_model import create_model, create_model_mtl_mtv_temperature, \
    create_model_mtl_mtv_exchange_rate
from kgp.metrics import root_mean_squared_error as RMSE
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import numpy as np

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import os

if __name__ == '__main__':

    time_step_lag = 6
    HORIZON = 1

    imfs_count = 11

    data_dir = 'data'
    # output_dir = 'output/exchange-rate/mtl/lag' + str(time_step_lag)
    output_dir = 'output/exchange-rate/mtl_mtv/horizon_' + str(HORIZON) + '/lag' + str(time_step_lag)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/model_checkpoint', exist_ok=True)

    multi_time_series = load_data_full(data_dir, datasource='exchange-rate', imfs_count=imfs_count, freq='d')
    print(multi_time_series.head())

    valid_start_dt = '2002-06-18'
    test_start_dt = '2006-08-13'

    features = ["load", "imf9", "imf10", "imf8", "imf7"]
    targets = ["load", "imf9", "imf10", "imf8", "imf7"]

    train_inputs, valid_inputs, test_inputs, y_scaler = split_train_validation_test(multi_time_series,
                                                     valid_start_time=valid_start_dt,
                                                     test_start_time=test_start_dt,
                                                     time_step_lag=time_step_lag,
                                                     horizon=HORIZON,
                                                     features=features,
                                                     target=targets,
                                                    time_format = '%Y-%m-%d',
                                                    freq = 'd'
                                                        )

    # ['imf6', 'imf5', 'imf4', 'imf3', 'imf2', 'imf0', 'imf1']
    aux_features = ["load", "imf6", "imf5", 'imf4', 'imf3', 'imf2', 'imf0', 'imf1']
    # for i in range(imfs_count):
    #     l = 'imf' + str(i)
    #     if l not in features:
    #         aux_features.append(l)

    aux_inputs, aux_valid_inputs, aux_test_inputs, aux_y_scaler = split_train_validation_test(multi_time_series,
                                                     valid_start_time=valid_start_dt,
                                                     test_start_time=test_start_dt,
                                                     time_step_lag=time_step_lag,
                                                     horizon=HORIZON,
                                                     features=aux_features,
                                                     target=["load"],
                                                    time_format = '%Y-%m-%d',
                                                    freq = 'd'
                                                     )

    X_train = train_inputs['X']
    y1_train = train_inputs['target_load']
    y2_train = train_inputs['target_imf9']
    y3_train = train_inputs['target_imf10']
    y4_train = train_inputs['target_imf8']
    y5_train = train_inputs['target_imf7']
    y_train = [y1_train, y2_train, y3_train, y4_train, y5_train]

    X_valid = valid_inputs['X']
    y1_valid = valid_inputs['target_load']
    y2_valid = valid_inputs['target_imf9']
    y3_valid = valid_inputs['target_imf10']
    y4_valid = valid_inputs['target_imf8']
    y5_valid = valid_inputs['target_imf7']
    y_valid = [y1_valid, y2_valid, y3_valid, y4_valid, y5_valid]

    aux_train = aux_inputs['X']
    aux_valid = aux_valid_inputs['X']
    aux_test = aux_test_inputs['X']

    # input_x = train_inputs['X']
    print("train_X shape", X_train.shape, "train_Y shape:", y1_train.shape)
    print("valid_X shape", X_valid.shape, "valid Y shape:", y1_valid.shape)
    print("aux_train shape", aux_train.shape, "aux valid Y shape", aux_valid.shape)
    # print("target shape", y_train.shape)
    # print("training size:", len(train_inputs['X']), 'validation', len(valid_inputs['X']), 'test size:', len(test_inputs['X']) )
    # print("sum sizes", len(train_inputs['X']) + len(valid_inputs['X']) + len(test_inputs['X']))

    # LATENT_DIM = 5
    BATCH_SIZE = 32
    EPOCHS = 30

    model = create_model_mtl_mtv_exchange_rate(horizon=HORIZON, nb_train_samples=len(X_train),
                                 batch_size=32, feature_count=len(features), lag_time=time_step_lag,
                                               aux_feature_count=len(aux_features))
    earlystop = EarlyStopping(monitor='loss', patience=5)

    file_path = output_dir + '/model_checkpoint/weights-improvement-{epoch:02d}.hdf5'
    check_point = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=True, mode='auto', period=1)

    history = model.fit([X_train, aux_train],
                        y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=([X_valid, aux_valid], y_valid),
              callbacks=[earlystop, check_point],
              verbose=1)

    store_training_loss(history=history, filepath=output_dir + "/training_loss_epochs_" + str(EPOCHS) + "_lag" +
                                                  str(time_step_lag) + ".csv")

    # Finetune the model
    # model.finetune(X_train, y_train, batch_size=BATCH_SIZE, gp_n_iter=10, verbose=1)

    # Test the model
    X_test = test_inputs['X']
    y1_test = test_inputs['target_load']
    y2_test = test_inputs['target_imf9']
    y3_test = test_inputs['target_imf10']
    y4_test = test_inputs['target_imf8']
    y5_test = test_inputs['target_imf7']

    y1_preds, y2_preds, y3_preds, y4_preds, y5_preds = model.predict([X_test, aux_test])
    # y1_preds, y2_preds, y3_preds, y4_preds = model.predict([X_test, aux_test])

    y1_test = y_scaler.inverse_transform(y1_test)
    y1_preds = y_scaler.inverse_transform(y1_preds)

    y1_test, y1_preds = flatten_test_predict(y1_test, y1_preds)

    rmse_predict = RMSE(y1_test, y1_preds)
    evs = explained_variance_score(y1_test, y1_preds)
    mae = mean_absolute_error(y1_test, y1_preds)
    mse = mean_squared_error(y1_test, y1_preds)
    msle = mean_squared_log_error(y1_test, y1_preds)
    meae = median_absolute_error(y1_test, y1_preds)
    r_square = r2_score(y1_test, y1_preds)

    mape_v = mape(y1_preds.reshape(-1, 1), y1_test.reshape(-1, 1))

    print('rmse_predict:', rmse_predict, "evs:", evs, "mae:", mae,
          "mse:", mse, "msle:", msle, "meae:", meae, "r2:", r_square, "mape", mape_v)

    store_predict_points(y1_test, y1_preds, output_dir + '/test_mtl_prediction_epochs_' + str(EPOCHS) + '_lag_'
                         + str(time_step_lag) + '.csv')