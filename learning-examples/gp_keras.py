from keras.layers import Input, SimpleRNN
from keras.optimizers import Adam

from kgp.layers import GP
from kgp.models import Model
from kgp.losses import gen_gp_loss


input_shape = (10, 2)  # 10 time steps, 2 dimensions
batch_size = 32
nb_train_samples = 512
gp_hypers = {'lik': -2.0, 'cov': [[-0.7], [0.0]]}

# Build the model
inputs = Input(shape=input_shape)
rnn = SimpleRNN(32)(inputs)
gp = GP(gp_hypers, batch_size=batch_size, nb_train_samples=nb_train_samples)
outputs = [gp(rnn)]
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
loss = [gen_gp_loss(gp) for gp in model.output_layers]
model.compile(optimizer=Adam(1e-2), loss=loss)

model.summary()