#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from collections import UserDict
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Image
get_ipython().magic(u'matplotlib inline')


# In[2]:


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


# In[5]:


T = 6
HORIZON = 1


# In[6]:


train = data.copy()[data.index < valid_start_dt][['avg_electricity']]
scaler = MinMaxScaler()


# In[7]:


train['avg_electricity'] = scaler.fit_transform(train)


# In[8]:


train_shifted = train.copy()
train_shifted['y_t+1'] = train_shifted['avg_electricity'].shift(-1, freq='H')


# In[9]:


for t in range(1, T+1):
    train_shifted['elec_t-'+str(T-t)] = train_shifted['avg_electricity'].shift(T-t, freq='H')
train_shifted = train_shifted.rename(columns={'avg_electricity':'elec_original'})
train_shifted.head(10)


# In[10]:


train_shifted = train_shifted.dropna(how='any')
train_shifted.head(5)


# In[11]:


y_train = train_shifted[['y_t+1']].as_matrix()
X_train = train_shifted[['elec_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()


# In[12]:


X_train = X_train.reshape(X_train.shape[0], T, 1)


# In[13]:


y_train.shape
y_train[:3]
X_train.shape
X_train[:3]
train_shifted.head(3)


# In[14]:


look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = data.copy()[(data.index >=look_back_dt) & (data.index < test_start_dt)][['avg_electricity']]


# In[15]:


valid['load'] = scaler.transform(valid)


# In[16]:


valid_shifted = valid.copy()
valid_shifted['y+1'] = valid_shifted['avg_electricity'].shift(-1, freq='H')
for t in range(1, T+1):
    valid_shifted['elec_t-'+str(T-t)] = valid_shifted['avg_electricity'].shift(T-t, freq='H')
valid_shifted = valid_shifted.dropna(how='any')


# In[17]:


y_valid = valid_shifted['y+1'].as_matrix()
X_valid = valid_shifted[['elec_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()


# In[18]:


X_valid = X_valid.reshape(X_valid.shape[0], T, 1)


# In[19]:


y_valid.shape


# In[20]:


y_valid[:3]
X_valid.shape
X_valid[:3]


# In[21]:


from keras.models import Model, Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping


# In[22]:


LATENT_DIM = 5 # number of units in the RNN layer
BATCH_SIZE = 32 # number of samples per mini-batch
EPOCHS = 10 # maximum number of times the training algorithm will cycle through all samples


# In[23]:


model = Sequential()
model.add(GRU(LATENT_DIM, input_shape=(T, 1)))
model.add(Dense(HORIZON))


# In[24]:


model.compile(optimizer='RMSprop', loss='mse')
model.summary()


# In[25]:


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)


# In[26]:


history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_valid, y_valid),
                    callbacks=[earlystop],
                    verbose=1)


# In[27]:


plot_df = pd.DataFrame.from_dict({'train_loss':history.history['loss'], 'val_loss':history.history['val_loss']})
plot_df.plot(logy=True, figsize=(10,10), fontsize=12)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.show()


# In[28]:


look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = data.copy()[test_start_dt:][['avg_electricity']]
test['avg_electricity'] = scaler.transform(test)


# In[29]:


test_shifted = test.copy()
test_shifted['y_t+1'] = test_shifted['avg_electricity'].shift(-1, freq='H')
for t in range(1, T+1):
    test_shifted['elec_t-'+str(T-t)] = test_shifted['avg_electricity'].shift(T-t, freq='H')
test_shifted = test_shifted.dropna(how='any')
y_test = test_shifted['y_t+1'].as_matrix()
X_test = test_shifted[['elec_t-'+str(T-t) for t in range(1, T+1)]].as_matrix()
X_test = X_test.reshape(X_test.shape[0], T, 1)


# In[33]:


y_test.shape


# In[34]:


X_test.shape


# In[35]:


predictions = model.predict(X_test)
predictions


# In[36]:


eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test_shifted.index
eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
eval_df['actual'] = np.transpose(y_test).ravel()
eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
eval_df.head()


# In[37]:


def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()


# In[38]:


mape(eval_df['prediction'], eval_df['actual'])


# In[39]:


eval_df[eval_df.timestamp<'2014-11-08'].plot(x='timestamp', y=['prediction', 'actual'], style=['r', 'b'], figsize=(15, 8))
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()

