from pandas import Series
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
from pandas.core.nanops import nanmean as pd_nanmean

import statsmodels.api as sm
import numpy as np
# df = pd.read_csv('../data/data-2011.csv', header=1, converters=)

def decompose(df, period=365, lo_frac=0.6, lo_delta=0.01):
    """Create a seasonal-trend (with Loess, aka "STL") decomposition of observed time series data.
    This implementation is modeled after the ``statsmodels.tsa.seasonal_decompose`` method
    but substitutes a Lowess regression for a convolution in its trend estimation.
    This is an additive model, Y[t] = T[t] + S[t] + e[t]
    For more details on lo_frac and lo_delta, see:
    `statsmodels.nonparametric.smoothers_lowess.lowess()`
    Args:
        df (pandas.Dataframe): Time series of observed counts. This DataFrame must be continuous (no
            gaps or missing data), and include a ``pandas.DatetimeIndex``.
        period (int, optional): Most significant periodicity in the observed time series, in units of
            1 observation. Ex: to accomodate strong annual periodicity within years of daily
            observations, ``period=365``.
        lo_frac (float, optional): Fraction of data to use in fitting Lowess regression.
        lo_delta (float, optional): Fractional distance within which to use linear-interpolation
            instead of weighted regression. Using non-zero ``lo_delta`` significantly decreases
            computation time.
    Returns:
        `statsmodels.tsa.seasonal.DecomposeResult`: An object with DataFrame attributes for the
            seasonal, trend, and residual components, as well as the average seasonal cycle.
    """
    # use some existing pieces of statsmodels
    lowess = sm.nonparametric.lowess
    _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)

    # get plain np array
    observed = np.asanyarray(df).squeeze()

    # calc trend, remove from observation
    trend = lowess(observed, [x for x in range(len(observed))],
                   frac=lo_frac,
                   delta=lo_delta * len(observed),
                   return_sorted=False)
    detrended = observed - trend

    # period must not be larger than size of series to avoid introducing NaNs
    period = min(period, len(observed))

    # calc one-period seasonality, remove tiled array from detrended
    period_averages = np.array([pd_nanmean(detrended[i::period]) for i in range(period)])
    # 0-center the period avgs
    period_averages -= np.mean(period_averages)
    seasonal = np.tile(period_averages, len(observed) // period + 1)[:len(observed)]
    resid = detrended - seasonal

    # convert the arrays back to appropriate dataframes, stuff them back into
    #  the statsmodel object
    results = list(map(_pandas_wrapper, [seasonal, trend, resid, observed]))
    dr = DecomposeResult(seasonal=results[0],
                         trend=results[1],
                         resid=results[2],
                         observed=results[3],
                         period_averages=period_averages)
    return dr

file = '../data/data-2011.csv'
headers = ['time', 'avg_electricity']
dtypes = {'time': 'str', 'avg_electricity': 'float'}
parse_dates = ['time']
df = pd.read_csv(file, sep=',', header=0, names=headers, dtype=dtypes, parse_dates=parse_dates)



# some hijinks to get around outdated statsmodels code

start = df['time'][0]
periods = len(df)
index = pd.date_range(start=start, end='2011-12-31 23:45:00', freq="15min")
# df.index = index

my_data = df['avg_electricity']
df = pd.DataFrame(data=my_data.values, index=index, columns=['col2'])

print(df.head(3))
# obs = pd.DataFrame(df['avg_electricity'], index=index, columns=['col2'])
#
# """Return packaged data in a pandas.DataFrame"""
# # some hijinks to get around outdated statsmodels code
# # dataset = sm.datasets.co2.load()
# # start = dataset.data['date'][0].decode('utf-8')
# # index = pd.date_range(start=start, periods=len(dataset.data), freq='W-SAT')
# # obs = pd.DataFrame(dataset.data['co2'], index=index, columns=['co2'])
# #


# x = np.asanyarray(df).squeeze()
# test_negative = np.any(x <= 0)
# negative_index = df[x <=0]

# interpolate for zero or negative values
df = (df
       .resample('D')
       .mean()
       .interpolate('linear'))

# result = seasonal_decompose(df, model='multiplicative', freq=35039)
result = seasonal_decompose(df, model='multiplicative')

seasonal_component = result.seasonal
trend_component = result.trend
residual_component = result.resid

reconstructed = pd.DataFrame(seasonal_component.values*trend_component.values*residual_component.values,
                             columns=seasonal_component.columns,
                             index=seasonal_component.index)


result.plot()
pyplot.show()


reconstructed.plot()
pyplot.show()