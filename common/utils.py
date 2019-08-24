import numpy as np
import pandas as pd
import os
from collections import UserDict

from pandas.core.indexes.datetimes import DatetimeIndex


def load_data(data_dir):
    """Load the GEFCom 2014 energy load data"""

    imf1 = pd.read_csv(os.path.join(data_dir, 'norm-2011-eIMF-1.csv'), header=None)

    imf2 = pd.read_csv(os.path.join(data_dir, 'norm-2011-eIMF-2.csv'), header=None)

    # Reindex the dataframe such that the dataframe has a record for every time point
    # between the minimum and maximum timestamp in the time series. This helps to
    # identify missing time periods in the data (there are none in this dataset).

    # energy.index = energy['timestamp']
    # energy = energy.reindex(pd.date_range(min(energy['timestamp']),
    #                                       max(energy['timestamp']),
    #                                       freq='H'))
    # energy = energy.drop('timestamp', axis=1)
    df = pd.concat([imf1, imf2], axis=1)
    df.columns = ["imf1", "imf2"]

    dt_idx = DatetimeIndex(freq='H', start='2011-01-01 00:00:00', end='2011-12-31 23:00:00')

    df.index = dt_idx

    return df


def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()


def create_evaluation_df(predictions, test_inputs, H, scaler):
    """Create a data frame for easy evaluation"""
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, H+1)])
    eval_df['timestamp'] = test_inputs.dataframe.index
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.transpose(test_inputs['target']).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    return eval_df