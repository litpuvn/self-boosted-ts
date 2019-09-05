from PyEMD import EEMD
from PyEMD import CEEMDAN

# import matplotlib as plt
import pylab as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats import pearsonr

if __name__ == "__main__":

    # data_source = 'elec'
    # data_source = 'temperature'
    data_source = 'exchange-rate'


    original_series = 'data/hourly_clean_electricity.csv'
    folder = 'data/electricity'

    if data_source == 'elec':
        original_series = 'data/hourly_clean_electricity.csv'
        folder = 'data/electricity'
        nIMFs = 13

    elif data_source == 'temperature':
        original_series = 'data/temperature.csv'
        folder = 'data/temperature'
        nIMFs = 12
    elif data_source == 'exchange-rate':
        original_series = 'data/time_exchange_rate.csv'
        folder = 'data/exchange-rate'
        nIMFs = 11
    else:
        raise Exception('bad data source:', data_source)


    eIMFs = []

    time_series_values = np.loadtxt(original_series, delimiter=',', usecols=1, skiprows=1)
    time_series_values = time_series_values[::2]

    for i in range(nIMFs):
        filepath = folder + '/imfs/IMF-' + str(i) + '.csv'
        imf = np.loadtxt(filepath, delimiter=',', usecols=0)

        imf = imf[::2]
        eIMFs.append(imf)

    time_values = np.linspace(0, 1, len(eIMFs[0]))

    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(time_values, time_series_values, 'r')
    plt.ylabel("Original")

    corr_data = []
    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(time_values, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        #
        # reduce number of ticks to 5
        plt.locator_params(axis='y', nbins=5)


    plt.xlabel("Time [s]")
    plt.tight_layout()
    # # plt.savefig('output/eemd_example', dpi=120)
    plt.show()

