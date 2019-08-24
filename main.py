from keras.callbacks import EarlyStopping

from common.TimeseriesTensor import TimeSeriesTensor
from common.utils import load_data, split_train_validation_test
from ts_model import create_model

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

# y_col = 'target'
# X_cols = []
# for i in range(1, time_step_lag+1):
#     for var_name in ["load", "imf1", "imf2"]:
#         X_cols = X_cols + [var_name + '-' + str(i)]
#
# # 5.Transform this Pandas dataframe into a numpy array
# X_train = train_inputs.dataframe[X_cols].as_matrix()
# y_train = train_inputs.dataframe[y_col].as_matrix()


X_train = train_inputs['X']
y_train = train_inputs['target']
# input_x = train_inputs['X']
print("input_X shape", X_train.shape)
print("target shape", y_train.shape)
print("training size:", len(train_inputs['X']), 'validation', len(valid_inputs['X']), 'test size:', len(test_inputs['X']) )
print("sum sizes", len(train_inputs['X']) + len(valid_inputs['X']) + len(test_inputs['X']))

LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 10

model = create_model()
earlystop = [EarlyStopping(monitor='val_mse', patience=10)]

model.fit(train_inputs['X'],
          train_inputs['target'],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(valid_inputs['X'], valid_inputs['target']),
          callbacks=[earlystop],
          verbose=1)


# train = energy.copy()[energy.index < valid_start_dt][['load']]

from sklearn.preprocessing import MinMaxScaler
# transforming data
# scaler = MinMaxScaler()
# scaler.fit(train[['load']])
# train[['load']] = scaler.transform(train)
#
# tensor_structure = {'X':(range(-time_step_lag+1, 1), ['load'])}
# train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)