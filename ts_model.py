from keras.callbacks import EarlyStopping
from keras.engine.topology import Input
from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten, Dense, Activation
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


def create_model(horizon=1, nb_train_samples=512, batch_size=32):

    x = Input(shape=(6, 11), name="input_layer")
    conv = Conv1D(kernel_size=3, filters=5, activation='relu')(x)
    mp = MaxPooling1D(pool_size=2)(conv)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)


    # sub2 = Dense(16, name="sub_task2")(shared_dense)
    # sub3 = Dense(16, name="sub_task3")(shared_dense)
    # sub3 = Flatten()(sub3)

    sub1 = GRU(units=16, name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)

    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(1, name="out1")(sub1)
    out2 = Dense(1, name="out2")(sub2)
    out3 = Dense(1, name="out3")(sub3)
    # Gaussian setting
    gp_hypers = {'lik': -2.0, 'cov': [[-0.7], [0.0]]}
    gp1 = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)
    gp2 = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)
    gp3 = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)

    outputs = [gp1(out1), gp2(out2), gp3(out3)]

    model = Model(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model

def create_model_mtl(horizon=1, nb_train_samples=512, batch_size=32):

    x = Input(shape=(6, 11), name="input_layer")
    conv = Conv1D(kernel_size=3, filters=5, activation='relu')(x)
    conv2 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=2)(conv)
    conv3 = Conv1D(5, kernel_size=3, padding='causal', strides=1, activation='relu', dilation_rate=4)(conv2)

    mp = MaxPooling1D(pool_size=2)(conv3)
    # conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    # mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = GRU(16, return_sequences=True)(mp)
    lstm2 = GRU(32, return_sequences=True)(lstm1)

    shared_dense = Dense(64, name="shared_layer")(lstm2)


    # sub2 = Dense(16, name="sub_task2")(shared_dense)
    # sub3 = Dense(16, name="sub_task3")(shared_dense)
    # sub3 = Flatten()(sub3)

    sub1 = GRU(units=16, name="task1")(shared_dense)
    sub2 = GRU(units=16, name="task2")(shared_dense)
    sub3 = GRU(units=16, name="task3")(shared_dense)

    # out1_gp = Dense(1, name="out1_gp")(sub1)
    out1 = Dense(1, name="out1")(sub1)
    out2 = Dense(1, name="out2")(sub2)
    out3 = Dense(1, name="out3")(sub3)

    outputs = [out1, out2, out3]

    model = KerasModel(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model