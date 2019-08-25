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


# In[3]:


data.head()


# In[4]:


data.index = data['time']


# In[5]:


data.head()


# In[6]:


data = data.reindex(pd.date_range(min(data['time']), max(data['time']), freq='H'))


# In[7]:


data.head()


# In[8]:


data = data.drop('time', axis=1)



# In[9]:


data.head()


# In[10]:


data = data[['avg_electricity']]
data.head()



# In[11]:


#validation and test start dates
valid_start_dt = '2011-10-01 0:00:00'
test_start_dt = '2011-11-01 0:00:00'

T = 6
HORIZON = 2


# In[12]:


data.dtypes


# In[13]:


train = data.copy()[data.index < valid_start_dt][['avg_electricity']]


# In[14]:


from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()
y_scaler.fit(train[['avg_electricity']])

X_scaler = MinMaxScaler()
train[['avg_electricity']] = X_scaler.fit_transform(train)


# In[15]:


class TimeSeriesTensor(UserDict):
    """A dictionary of tensors for input into the RNN model.
    
    Use this class to:
      1. Shift the values of the time series to create a Pandas dataframe containing all the data
         for a single training example
      2. Discard any samples with missing values
      3. Transform this Pandas dataframe into a numpy array of shape 
         (samples, time steps, features) for input into Keras

    The class takes the following parameters:
       - **dataset**: original time series
       - **target** name of the target column
       - **H**: the forecast horizon
       - **tensor_structures**: a dictionary discribing the tensor structure of the form
             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
             if features are non-sequential and should not be shifted, use the form
             { 'tensor_name' : (None, [feature, feature, ...])}
       - **freq**: time series frequency (default 'H' - hourly)
       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
    """
    
    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())
        
        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)
    
    def _shift_data(self, H, freq, drop_incomplete):
        
        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()
        
        idx_tuples = []
        for t in range(1, H+1):
            df['t+'+str(t)] = df[self.target].shift(t*-1, freq=freq)
            idx_tuples.append(('target', 'y', 't+'+str(t)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]
            
            for col in dataset_cols:
            
            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))
                
        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df
    
    def _df2tensors(self, dataframe):
        
        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']
    
        inputs = {}
        y = dataframe['target']
        y = y.as_matrix()
        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].as_matrix()
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs
       
    def subset_data(self, new_dataframe):
        
        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.
        
        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)


# In[16]:


tensor_structure = {'X':(range(-T+1, 1), ['avg_electricity'])}
train_inputs = TimeSeriesTensor(train, 'avg_electricity', HORIZON, {'X':(range(-T+1, 1), ['avg_electricity'])})


# In[17]:


train_inputs.dataframe.head()


# In[18]:


look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = data.copy()[(data.index >=look_back_dt) & (data.index < test_start_dt)][['avg_electricity']]
valid['avg_electricity'] = X_scaler.transform(valid)


# In[19]:


valid_inputs = TimeSeriesTensor(valid, 'avg_electricity', HORIZON, tensor_structure)


# In[20]:


from keras.models import Model, Sequential
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping


# In[21]:


LATENT_DIM = 5
BATCH_SIZE = 32
EPOCHS = 10


# In[22]:


model = Sequential()
model.add(GRU(LATENT_DIM, input_shape=(T, 1)))
model.add(RepeatVector(HORIZON))
model.add(GRU(LATENT_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.add(Flatten())


# In[23]:


model.compile(optimizer='RMSprop', loss='mse')


# In[24]:


model.summary()


# In[25]:


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)


# In[26]:


model.fit(train_inputs['X'],
          train_inputs['target'],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(valid_inputs['X'], valid_inputs['target']),
          callbacks=[earlystop],
          verbose=1)


# In[ ]:





# In[30]:


look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = data.copy()[test_start_dt:][['avg_electricity']]
test[['avg_electricity']] = X_scaler.transform(test)
test_inputs = TimeSeriesTensor(test, 'avg_electricity', HORIZON, tensor_structure)


# In[31]:


predictions = model.predict(test_inputs['X'])


# In[32]:


predictions


# In[34]:


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df


# In[35]:


eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, y_scaler)
eval_df.head()


# In[36]:


eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
eval_df.groupby('h')['APE'].mean()


# In[38]:


def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()


# In[39]:


mape(eval_df['prediction'], eval_df['actual'])


# In[41]:


eval_df[eval_df.timestamp<'2014-11-08'].plot(x='timestamp', y=['prediction', 'actual'], style=['r', 'b'], figsize=(20, 8))
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('avg_electricity', fontsize=12)
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(eval_df['actual'], eval_df['prediction']))
print(rmse)
