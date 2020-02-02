from random import randrange
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from pandas import DataFrame, DatetimeIndex
import pandas as pd

file_configs = [
    {
        'path': 'data/temperature.csv',
        'start': '2004-03-10 18:00:00',
        'freq': 'h',
        'output': 'seasonal_temperature.csv'
    },
    {
        'path': 'data/hourly_clean_electricity.csv',
        'start': '2011-01-01 00:00:00',
        'freq': 'h',
        'output': 'seasonal_electricity.csv'
    },
    {
        'path': 'data/time_exchange_rate.csv',
        'start': '1/1/1990',
        'freq': 'd',
        'output': 'seasonal_exchange_rate.csv'
    },
]

def convert_to_seaonal(config, save=False, show_graph=False, model='additive'):
    filepath = config['path']
    start = config['start']
    freq = config['freq']
    output = config['output']

    if filepath is None or start is None or freq is None or output is None:
        raise Exception('Bad configuration')

    folder = 'data/seasonal'
    # time_series_values = np.loadtxt(filepath, delimiter=',', usecols=1, skiprows=1, converters = {1: to_float})
    time_series_values = np.loadtxt(filepath, delimiter=',', usecols=1, skiprows=1)
    time_values = np.linspace(0, 1, len(time_series_values))
    data = DataFrame(time_series_values, DatetimeIndex(start=start,
                                      periods=len(time_series_values),
                                      freq=freq))
    result = seasonal_decompose(data, model=model, extrapolate_trend='freq')

    observed = result.observed
    resid = result.resid
    seasonal = result.seasonal
    trend = result.trend

    horizontal_stack = pd.concat([observed, resid, seasonal, trend], axis=1, ignore_index=True)
    horizontal_stack.columns = ['Observed', 'Residual', 'Seasonal', 'Trend']
    if save == True:
        horizontal_stack.to_csv(folder + '/' + model + '_' + output, index=True)
    else:
        print(horizontal_stack.head(5))

    if show_graph == True:
        result.plot()
        pyplot.show()


for config in file_configs:
    convert_to_seaonal(config=config, save=True, show_graph=False, model='additive')
    convert_to_seaonal(config=config, save=True, show_graph=False, model='multiplicative')