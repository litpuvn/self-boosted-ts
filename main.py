from keras.callbacks import EarlyStopping
import pandas as pd
from common.TimeseriesTensor import TimeSeriesTensor
from common.utils import load_data, split_train_validation_test
from ts_model import create_model
from kgp.metrics import root_mean_squared_error as RMSE
import matplotlib.pyplot as plt

time_step_lag = 6
HORIZON = 3

data_dir = 'data/'
multi_time_series = load_data(data_dir)
print(multi_time_series.head())


valid_start_dt = '2011-09-01 00:00:00'
test_start_dt = '2011-11-01 00:00:00'

train_inputs, valid_inputs, test_inputs = split_train_validation_test(multi_time_series,
                                                 valid_start_time=valid_start_dt,
                                                 test_start_time=test_start_dt,
                                                 time_step_lag=time_step_lag,

                                                 features=["load", "imf1", "imf2"]
                                                 )


X_train = train_inputs['X']
y1_train = train_inputs['target_load']
y2_train = train_inputs['target_imf1']
y3_train = train_inputs['target_imf2']


X_valid = valid_inputs['X']
y1_valid = valid_inputs['target_load']
y2_valid = valid_inputs['target_imf1']
y3_valid = valid_inputs['target_imf2']


# input_x = train_inputs['X']
print("input_X shape", X_train.shape)
# print("target shape", y_train.shape)
# print("training size:", len(train_inputs['X']), 'validation', len(valid_inputs['X']), 'test size:', len(test_inputs['X']) )
# print("sum sizes", len(train_inputs['X']) + len(valid_inputs['X']) + len(test_inputs['X']))

# LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 1

model = create_model(nb_train_samples=len(X_train), batch_size=32)
earlystop = EarlyStopping(monitor='val_mse', patience=10)

history = model.fit(X_train,
          [y1_train, y2_train, y3_train],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(X_valid, [y1_valid, y2_valid, y3_valid]),
          callbacks=[earlystop],
          verbose=1)


plot_df = pd.DataFrame.from_dict({'train_loss': history.history['loss'], 'val_loss': history.history['val_loss']})
plot_df.plot(logy=True, figsize=(10, 10), fontsize=12)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.show()


# Finetune the model
model.finetune(*X_train, batch_size=BATCH_SIZE, gp_n_iter=100, verbose=1)

# Test the model
X_test = test_inputs['X']
y1_test = test_inputs['target_load']
y2_test = test_inputs['target_imf1']
y3_test = test_inputs['target_imf2']

y1_preds, y2_preds, y3_preds = model.predict(X_test)

rmse_predict = RMSE(y1_test, y1_preds)
print('Test predict RMSE:', rmse_predict)