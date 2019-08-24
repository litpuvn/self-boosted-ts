from keras.callbacks import EarlyStopping
from keras.engine.topology import Input
from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import Adam

from kgp.layers import GP
from kgp.models import Model
from kgp.losses import gen_gp_loss
from kgp.utils.experiment import train


def create_model():
    n = 19  # n features
    x = Input(shape=(None, 3))

    conv = Conv1D(filters=5, kernel_size=3, activation='relu')(x)
    mp = MaxPooling1D(pool_size=2)(conv)
    conv2 = Conv1D(filters=5, kernel_size=3, activation='relu')(mp)
    mp = MaxPooling1D(pool_size=2)(conv2)

    lstm1 = LSTM(50, return_sequences=True)(mp)
    lstm2 = LSTM(10, return_sequences=True)(lstm1)

    shared_dense = Dense(16)(lstm2)


    sub2 = Dense(16)(shared_dense)
    sub3 = Dense(16)(shared_dense)


    out1 = SimpleRNN(units=32)(shared_dense)


    out2 = Dense(1)(sub2)
    out3 = Dense(1)(sub3)

    # Gaussian setting
    batch_size = 32
    nb_train_samples = 512
    gp_hypers = {'lik': -2.0, 'cov': [[-0.7], [0.0]]}
    gp = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)

    outputs = [gp(out1), out2, out3]

    model = Model(inputs=x, outputs=outputs)


    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'], loss_weights=[0.5, 0.25, 0.25])
    # Callbacks
    # callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    model.summary()

    return model