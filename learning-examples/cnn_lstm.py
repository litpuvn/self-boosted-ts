from keras.engine.topology import Input
from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten, Dense
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=5, kernel_size=3, activation='relu'), batch_input_shape=(24, None, 24, 1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, stateful=True, return_sequences=True))
model.add(LSTM(10, stateful=True, return_sequences=True))
model.add(Dense(24))

model.summary()

# model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'acc'])
#
#
# n = 10  # n features
# x = Input(shape=(n, 24, ))
# td1 = TimeDistributed(Conv1D(filters=5, kernel_size=3, activation='relu')(x))
# mxpooling = MaxPooling1D(pool_size=2)(td1)
# td2 = TimeDistributed(mxpooling)
# td3 = TimeDistributed(Flatten()(td2))
# lstm1 = LSTM(50, stateful=True, return_sequences=True)(td3)
# lstm2 = LSTM(10, stateful=True, return_sequences=True)(lstm1)