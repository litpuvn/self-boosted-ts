from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Dense

n = 10 # n features
x = Input(shape=(n, ))
shared = Dense(32)(x)
sub1 = Dense(16)(shared)
sub2 = Dense(16)(shared)
sub3 = Dense(16)(shared)
out1 = Dense(1)(sub1)
out2 = Dense(1)(sub2)
out3 = Dense(1)(sub3)

model = Model(inputs=x, outputs=[out1, out2, out3])

model.summary()
