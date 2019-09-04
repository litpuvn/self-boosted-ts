import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt

from pandas import DatetimeIndex
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from common.utils import mape

def RMSE(x):
    return sqrt(x)

if __name__ == '__main__':
    time_step_lag = 1
    HORIZON = 1

    data = pd.read_csv('/home/ope/Documents/Projects/self-boosted-ts/data/temperature.csv', parse_dates=['Date_Time'])
    data.index = data['Date_Time']
    data = data.reindex(pd.date_range(min(data['Date_Time']), max(data['Date_Time']), freq='H'))
    data = data.drop('Date_Time', axis=1)

    data = data[['temperature']]

    series = data




    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    model = ARIMA(history, order=(time_step_lag, 1, 0))
    model_fit = model.fit(disp=0)
    for index, t in enumerate(range(len(test))):

        if index % 10 == 0:
            print("rebuilding the model", index)
            model = ARIMA(history, order=(time_step_lag, 1, 0))
            model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()

    mse = mean_squared_error(test, predictions)
    rmse_predict = RMSE(mse)
    evs = explained_variance_score(test, predictions)
    mae = mean_absolute_error(test, predictions)

    meae = median_absolute_error(test, predictions)

    predictions = np.array(predictions)
    r_square = r2_score(test, predictions)
    mape_v = mape(predictions.reshape(-1, 1), test.reshape(-1, 1))

    print("mse:", mse, 'rmse_predict:', rmse_predict, "mae:", mae, "mape:", mape_v, "r2:", r_square,
          "meae:", meae, "evs:", evs)

