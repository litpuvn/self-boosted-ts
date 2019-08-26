from keras.callbacks import EarlyStopping
import pandas as pd
from common.TimeseriesTensor import TimeSeriesTensor
from common.gp_log import store_training_loss, store_predict_points, flatten_test_predict
from common.utils import load_data, split_train_validation_test
from ts_model import create_model
from kgp.metrics import root_mean_squared_error as RMSE
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

if __name__ == '__main__':

    time_step_lag = 6
    HORIZON = 1

    data_dir = 'data/'
    multi_time_series = load_data(data_dir)
    print(multi_time_series.head())


    valid_start_dt = '2011-09-01 00:00:00'
    test_start_dt = '2011-11-01 00:00:00'

    train_inputs, valid_inputs, test_inputs, y_scaler = split_train_validation_test(multi_time_series,
                                                     valid_start_time=valid_start_dt,
                                                     test_start_time=test_start_dt,
                                                     time_step_lag=time_step_lag,
                                                     horizon=HORIZON,
                                                     features=["load", "imf1", "imf2"],
                                                     target=["load", "imf1", "imf2"]
                                                     )

    X_train = train_inputs['X']
    y1_train = train_inputs['target_load']
    y2_train = train_inputs['target_imf1']
    y3_train = train_inputs['target_imf2']
    y_train = [y1_train, y2_train, y3_train]

    X_valid = valid_inputs['X']
    y1_valid = valid_inputs['target_load']
    y2_valid = valid_inputs['target_imf1']
    y3_valid = valid_inputs['target_imf2']
    y_valid = [y1_valid, y2_valid, y3_valid]

    # input_x = train_inputs['X']
    print("train_X shape", X_train.shape)
    print("valid_X shape", X_valid.shape)
    # print("target shape", y_train.shape)
    # print("training size:", len(train_inputs['X']), 'validation', len(valid_inputs['X']), 'test size:', len(test_inputs['X']) )
    # print("sum sizes", len(train_inputs['X']) + len(valid_inputs['X']) + len(test_inputs['X']))

    # LATENT_DIM = 5
    BATCH_SIZE = 32
    EPOCHS = 100

    model = create_model(horizon=HORIZON, nb_train_samples=len(X_train), batch_size=32)
    earlystop = EarlyStopping(monitor='val_mse', patience=5)

    file_path = 'output/model_checkpoint/weights-improvement-{epoch:02d}.hdf5'
    check_point = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=False,
                                  save_weights_only=True, mode='auto', period=1)

    model.load_weights('output/model_checkpoint/weights-improvement-10.hdf5')
    history = model.fit(X_train,
                        y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              initial_epoch=10,
              validation_data=(X_valid, y_valid),
              callbacks=[earlystop, check_point],
              verbose=1)

    store_training_loss(history=history, filepath="output/training_loss_epochs_" + str(EPOCHS) + ".csv")

    # Finetune the model
    model.finetune(X_train, y_train, batch_size=BATCH_SIZE, gp_n_iter=10, verbose=1)

    # Test the model
    X_test = test_inputs['X']
    y1_test = test_inputs['target_load']
    y2_test = test_inputs['target_imf1']
    y3_test = test_inputs['target_imf2']

    y1_preds, y2_preds, y3_preds = model.predict(X_test)

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

    print('rmse_predict:', rmse_predict, "evs:", evs, "mae:", mae,
          "mse:", mse, "msle:", msle, "meae:", meae, "r2:", r_square)

    store_predict_points(y1_test, y1_preds, 'output/test_mtl_prediction_epochs_' + str(EPOCHS) + '.csv')