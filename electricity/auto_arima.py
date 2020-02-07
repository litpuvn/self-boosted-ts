import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def arimamodel(timeseries):
    # automodel = pm.auto_arima(timeseries,
    #                           start_p=1,
    #                           start_q=1,
    #                           test="adf",
    #                           seasonal=False,
    #                           trace=True)
    automodel = pm.auto_arima(timeseries,
                              start_p=1,
                              start_q=1,
                              test="adf",
                              seasonal=True,
                              m=24,
                              trace=True)
    return automodel


def plotarima(n_periods, timeseries, automodel):
    # Forecast
    fc, confint = automodel.predict(n_periods=n_periods,
                                    return_conf_int=True)
    # Weekly index
    fc_ind = pd.date_range(timeseries.index[timeseries.shape[0]-1],
                           periods=n_periods, freq="W")
    # Forecast series
    fc_series = pd.Series(fc, index=fc_ind)
    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries)
    plt.plot(fc_series, color="red")
    plt.xlabel("date")
    plt.ylabel(timeseries.name)
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color="k",
                     alpha=0.25)
    plt.legend(("past", "forecast", "95% confidence interval"),
               loc="upper left")
    plt.show()


df = pd.read_csv("../data/seasonal/additive_seasonal_electricity.csv", sep=',', header=0, index_col=0, parse_dates=True)
# df.rename(columns={'Observed': 'load'}, inplace=True)

automodel = arimamodel(df["Seasonal"])

plotarima(70, df["Seasonal"], automodel)

print(automodel.summary())
