from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten, Dense
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from torch.nn.modules.rnn import LSTM

model = Sequential()

conv = Conv1D(filter=5, kernel_size=3, activation='relu', batch_input_shape=(24, None, 1))
model.add(TimeDistributed(conv))

maxpool = MaxPooling1D(pool_size=2)
model.add(TimeDistributed(maxpool))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, stateful=True, return_sequence=True))
model.add(LSTM(10, stateful=True, return_sequence=True))
model.add(Dense(24))


model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'acc'])


