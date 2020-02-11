from keras.callbacks import EarlyStopping
from keras.engine.topology import Input
from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.optimizers import Adam

from kgp.layers import GP
from kgp.models import Model

from kgp.losses import gen_gp_loss
from kgp.utils.experiment import train
from keras.models import Model as KerasModel
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate


def create_model(horizon=1, nb_train_samples=512, batch_size=32, feature_count=11):

    x = Input(shape=(6, feature_count), name="input_layer")
    conv = Conv1D(kernel_size=3, filters=5, activation='relu', dilation_rate=1)(x)
    conv2 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)
    mp = MaxPooling1D(pool_size=2)(conv3)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)
    shared_dense = Flatten()(shared_dense)
    sub1 = Dense(16, name="task1")(shared_dense)
    # sub2 = Dense(16, name="task2")(shared_dense)
    # sub3 = Dense(16, name="task3")(shared_dense)


    # sub1 = GRU(units=16, name="task1")(shared_dense)
    # sub2 = GRU(units=16, name="task2")(shared_dense)
    # sub3 = GRU(units=16, name="task3")(shared_dense)

    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(1, name="out1")(sub1)
    # out2 = Dense(1, name="out2")(sub2)
    # out3 = Dense(1, name="out3")(sub3)
    # Gaussian setting
    gp_hypers = {'lik': -2.0, 'cov': [[-0.7], [0.0]]}
    gp_params = {
        'cov': 'SEiso',
        'hyp_lik': -2.0,
        'hyp_cov': [[-0.7], [0.0]],
        'opt': {'cg_maxit': 500, 'cg_tol': 1e-4},
        'grid_kwargs': {'eq': 1, 'k': 1e2},
        'update_grid': True,
    }
    gp1 = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)
    # gp2 = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)
    # gp3 = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)

    outputs = [gp1(out1)]

    model = Model(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model

def create_model_mtl_only_electricity(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, time_lag=6):
    x = Input(shape=(time_lag, feature_count), name="input_layer")
    conv = Conv1D(kernel_size=1, filters=5, activation='relu')(x)
    conv2 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=1)(conv3)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=72, name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)

    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    outputs = [out1, out2, out3]

    model = KerasModel(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model



def create_model_mtv_electricity(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, time_lag=6):

    x = Input(shape=(time_lag, feature_count), name="input_layer")
    conv = Conv1D(kernel_size=1, filters=5, activation='relu')(x)
    conv2 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=1)(conv3)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=72, name="task1")(shared_dense)


    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)

    outputs = out1

    model = KerasModel(inputs=x, outputs=outputs)

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    # x = Input(shape=(time_lag, 12), name='aux_input')
    #
    # out1 = Dense(8, name="spec_out1")(x)
    # out1 = Flatten()(out1)
    # out1 = Dense(1, name="out1")(out1)
    #
    # outputs = out1
    #
    # model = KerasModel(inputs=x, outputs=outputs)
    #
    #
    # model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    # # Callbacks
    # # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]
    #
    # model.summary()
    #
    return model


def create_model_mtl_mtv_electricity(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, lag_time=6, auxiliary_feature_count=12):

    x = Input(shape=(lag_time, feature_count), name="input_layer")
    # conv = Conv1D(filters=5, kernel_size=3, activation='relu')(x)
    # conv = Conv1D(filters=1, kernel_size=1, activation='relu')(x)
    conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(x)
    # conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv)

    # mp = MaxPooling1D(pool_size=2)(conv)
    mp = MaxPooling1D(pool_size=1)(conv)


    lstm1 = GRU(16, return_sequences=True)(mp)
    # lstm2 = GRU(32, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=(lag_time*auxiliary_feature_count), name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)

    sub1 = Reshape((lag_time, auxiliary_feature_count))(sub1)
    auxiliary_input = Input(shape=(lag_time, auxiliary_feature_count), name='aux_input')

    concate = Concatenate(axis=-1)([sub1, auxiliary_input])
    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(8, name="spec_out1")(concate)
    out1 = Flatten()(out1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    outputs = [out1, out2, out3]
    # outputs = [out1, out2]

    model = KerasModel(inputs=[x, auxiliary_input], outputs=outputs)

    # adam optimizsor is good
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'], loss_weights=[0.5, 0.01, 0.01])
    # model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape', 'mae'], loss_weights=[0.5, 0.5, 0.5])
    # model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model



def create_model_mtl_only_temperature(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, time_lag=6):

    x = Input(shape=(time_lag, feature_count), name="input_layer")
    conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    conv2 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=1)(conv3)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=60, name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)
    sub4 = GRU(units=16, name="task4")(shared_dense)

    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    out4 = Dense(1, name="spec_out4")(sub4)
    out4 = Dense(1, name="out4")(out4)

    outputs = [out1, out2, out3, out4]

    model = KerasModel(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model



def create_model_mtv_temperature(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, lag_time=6):

    x = Input(shape=(lag_time, feature_count), name="input_layer")
    conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    conv2 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=1)(conv3)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=60, name="task1")(shared_dense)

    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)

    outputs = out1

    model = KerasModel(inputs=x, outputs=outputs)

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model


def create_model_mtl_mtv_temperature(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, time_lag=6, aux_feature_count=10):

    x = Input(shape=(time_lag, feature_count), name="input_layer")
    # conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    # conv2 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    # conv3 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)


    conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    # conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(x)
    # conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv)

    # mp = MaxPooling1D(pool_size=2)(conv)
    mp = MaxPooling1D(pool_size=1)(conv)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)
    # lstm2 = GRU(32, return_sequences=True)(mp)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=(time_lag*aux_feature_count), name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)
    sub4 = GRU(units=16, name="task4")(shared_dense)

    sub1 = Reshape((time_lag, aux_feature_count))(sub1)
    auxiliary_input = Input(shape=(time_lag, aux_feature_count), name='aux_input')

    concate = Concatenate(axis=-1)([sub1, auxiliary_input])
    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(8, name="spec_out1")(concate)
    out1 = Flatten()(out1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)
    #
    out4 = Dense(1, name="spec_out4")(sub4)
    out4 = Dense(1, name="out4")(out4)

    outputs = [out1, out2, out3, out4]

    model = KerasModel(inputs=[x, auxiliary_input], outputs=outputs)


    model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.01, 0.01, 0.01])
    # model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
    # model.compile(optimizer='SGD', loss='mse', metrics=['mae', 'mape', 'mse'])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model



def create_model_mtl_only_exchange_rate(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=6, time_lag=6):

    x = Input(shape=(time_lag, feature_count), name="input_layer")
    conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    conv2 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=1)(conv3)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=48, name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)
    sub4 = GRU(units=16, name="task4")(shared_dense)
    sub5 = GRU(units=16, name="task5")(shared_dense)

    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    out4 = Dense(1, name="spec_out4")(sub4)
    out4 = Dense(1, name="out4")(out4)

    out5 = Dense(1, name="spec_out5")(sub5)
    out5 = Dense(1, name="out5")(out5)

    outputs = [out1, out2, out3, out4, out5]

    model = KerasModel(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model



def create_model_mtv_exchange_rate(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=6, time_lag=6):
    x = Input(shape=(time_lag, feature_count), name="input_layer")
    conv = Conv1D(filters=5, kernel_size=3, activation='relu')(x)
    conv2 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=2)(conv3)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=48, name="task1")(shared_dense)


    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)


    outputs = out1

    model = KerasModel(inputs=x, outputs=outputs)

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'],
                  )
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model


def create_model_mtl_mtv_exchange_rate(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=6, lag_time=6, aux_feature_count=8):

    x = Input(shape=(lag_time, feature_count), name="input_layer")
    conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    # conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(x)
    conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv)

    mp = MaxPooling1D(pool_size=1)(conv)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(8, return_sequences=True)(mp)
    lstm2 = GRU(16, return_sequences=True)(lstm1)
    # lstm2 = GRU(16, return_sequences=True)(mp)

    shared_dense = Dense(32, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=(lag_time*aux_feature_count), name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)
    sub4 = GRU(units=16, name="task4")(shared_dense)
    sub5 = GRU(units=16, name="task5")(shared_dense)

    sub1 = Reshape((lag_time, aux_feature_count))(sub1)
    auxiliary_input = Input(shape=(lag_time, aux_feature_count), name='aux_input')

    concate = Concatenate(axis=-1)([sub1, auxiliary_input])
    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(8, name="spec_out1")(concate)
    out1 = Flatten()(out1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    out4 = Dense(1, name="spec_out4")(sub4)
    out4 = Dense(1, name="out4")(out4)

    out5 = Dense(1, name="spec_out5")(sub5)
    out5 = Dense(1, name="out5")(out5)

    outputs = [out1, out2, out3, out4, out5]
    # outputs = [out1, out2, out3, out4]

    model = KerasModel(inputs=[x, auxiliary_input], outputs=outputs)


    model.compile(optimizer='sgd', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[1, 0.00, 0.00, 0.00, 0.00])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model


def create_seasonal_model_mtl(horizon=1, nb_train_samples=512, batch_size=32,  feature_count=11, time_lag=6):

    x = Input(shape=(time_lag, feature_count), name="input_layer")
    conv = Conv1D(filters=5, kernel_size=1, activation='relu')(x)
    # conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(x)
    conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv = Conv1D(filters=5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv)

    mp = MaxPooling1D(pool_size=1)(conv)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)

    ## sub1 is main task; units = reshape dimension multiplication
    sub1 = GRU(units=60, name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)
    sub4 = GRU(units=16, name="task4")(shared_dense)

    out1 = Dense(8, name="spec_out1")(sub1)
    out1 = Dense(1, name="out1")(out1)

    out2 = Dense(8, name="spec_out2")(sub2)
    out2 = Dense(1, name="out2")(out2)

    out3 = Dense(1, name="spec_out3")(sub3)
    out3 = Dense(1, name="out3")(out3)

    out4 = Dense(1, name="spec_out4")(sub4)
    out4 = Dense(1, name="out4")(out4)

    outputs = [out1, out2, out3, out4]

    model = KerasModel(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model
