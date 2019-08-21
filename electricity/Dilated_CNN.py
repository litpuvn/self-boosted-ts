#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

data = pd.read_csv('data-2011.csv', parse_dates=['time'])
print (data.head())
print ('\n Data Types:')
print (data.dtypes)


data = pd.read_csv('data-2011.csv', parse_dates=['time'])
data.index = data['time']
data = data.reindex(pd.date_range(min(data['time']), max(data['time']), freq='H'))
data = data.drop('time', axis=1)

data = data[['avg_electricity']]
data.head()

data.dtypes

#validation and test start dates
valid_start_dt = '2011-10-01 0:00:00'
test_start_dt = '2011-11-01 0:00:00'

#separate and plot the three data
data[data.index < valid_start_dt][['avg_electricity']].rename(columns={'avg_electricity':'train'})     .join(data[(data.index >=valid_start_dt) & (data.index < test_start_dt)][['avg_electricity']]           .rename(columns={'avg_electricity':'validation'}), how='outer')     .join(data[test_start_dt:][['avg_electricity']].rename(columns={'avg_electricity':'test'}), how='outer')     .plot(y=['train', 'validation', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('time', fontsize=12)
plt.ylabel('avg_electricity', fontsize=12)
plt.show()



T = 10
HORIZON = 1


train = data.copy()[data.index < valid_start_dt][['avg_electricity']]



train.plot.hist(bins=100, fontsize=12)



#shifting the avg_electricity variable one hour in time
train_shifted = train.copy()
train_shifted['y_t+1'] = train_shifted['avg_electricity'].shift(-1, freq='H')
train_shifted.head(10)


for t in range(1, T+1):
    train_shifted['elec_t-'+str(T-t)] = train_shifted['avg_electricity'].shift(T-t, freq='H')
train_shifted = train_shifted.rename(columns={'avg_electricity':'elec_original'})
train_shifted.head(10)




#discard missing values
train_shifted = train_shifted.dropna(how='any')
train_shifted.head(5)


#convert target variable into a numpy array
y_train = train_shifted[['y_t+1']].as_matrix()




y_train.shape

y_train[:3]


X_train = train_shifted[['elec_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()
X_train = X_train[... , np.newaxis]


X_train.shape


X_train[:3]


train_shifted.head(3)

look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = data.copy()[(data.index >=look_back_dt) & (data.index < test_start_dt)][['avg_electricity']]
valid.head()



valid_shifted = valid.copy()
valid_shifted['y+1'] = valid_shifted['avg_electricity'].shift(-1, freq='H')
for t in range(1, T+1):
    valid_shifted['elec_t-'+str(T-t)] = valid_shifted['avg_electricity'].shift(T-t, freq='H')
valid_shifted = valid_shifted.dropna(how='any')
y_valid = valid_shifted['y+1'].as_matrix()
X_valid = valid_shifted[['elec_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()
X_valid = X_valid[..., np.newaxis]

y_valid.shape


X_valid.shape



from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint




LATENT_DIM = 5
KERNEL_SIZE = 2
BATCH_SIZE = 32
EPOCHS = 10


model = Sequential()
model.add(Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=1, input_shape=(T, 1)))
model.add(Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=2))
model.add(Conv1D(LATENT_DIM, kernel_size=KERNEL_SIZE, padding='causal', strides=1, activation='relu', dilation_rate=4))
model.add(Flatten())
model.add(Dense(HORIZON, activation='linear'))



model.summary()



model.compile(optimizer='Adam', loss='mse')



earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
best_val = ModelCheckpoint('model_{epoch:02d}.h5', save_best_only=True, mode='min', period=1)
history = model.fit(X_train,
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(X_valid, y_valid),
          callbacks=[earlystop, best_val],
          verbose=1)


best_epoch = np.argmin(np.array(history.history['val_loss']))+1
model.load_weights("model_{:02d}.h5".format(best_epoch))




plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})
plot_df.plot(logy=True, figsize=(10,10), fontsize=12)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.show()





#create the test set
look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = data.copy()[test_start_dt:][['avg_electricity']]
test.head()





#test set features

test_shifted = test.copy()
test_shifted['y_t+1'] = test_shifted['avg_electricity'].shift(-1, freq='H')
for t in range(1, T+1):
    test_shifted['elec_t-'+str(T-t)] = test_shifted['avg_electricity'].shift(T-t, freq='H')
test_shifted = test_shifted.dropna(how='any')
y_test = test_shifted['y_t+1'].as_matrix()
X_test = test_shifted[['elec_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()
X_test = X_test[... , np.newaxis]


predictions = model.predict(X_test)
predictions



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train['avg_electricity'] = scaler.fit_transform(train)
valid['avg_electricity'] = scaler.transform(valid)
test['load'] = scaler.transform(test)

eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test_shifted.index
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
eval_df['actual'] = np.transpose(y_test).ravel()
eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
eval_df.head()

def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()





mape(eval_df['prediction'], eval_df['actual'])





eval_df[eval_df.timestamp<'2014-11-08'].plot(x='timestamp', y=['prediction', 'actual'], style=['r', 'b'], figsize=(15, 8))
plt.xlabel('time', fontsize=12)
plt.ylabel('electricity', fontsize=12)
plt.show()







