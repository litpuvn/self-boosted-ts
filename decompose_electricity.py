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

    filepath = 'data/time_exchange_rate.csv'
    folder = 'data/exchange-rate'

    # time_series_values = np.loadtxt(filepath, delimiter=',', usecols=1, skiprows=1, converters = {1: to_float})
    time_series_values = np.loadtxt(filepath, delimiter=',', usecols=1, skiprows=1)

    time_values = np.linspace(0, 1, len(time_series_values))

    # reconstructed_points = np.sum(eIMFs, axis=0)

    # Execute EEMD on S
    # eemd = CEEMDAN(trials=100, epsilon=0.005, ext_EMD=None)
    # eIMFs = eemd.ceemdan(S=time_series_values, T=time_values, max_imf=-1)
    eemd = EEMD(trials=100, epsilon=0.005, ext_EMD=None)
    eIMFs = eemd.eemd(S=time_series_values, T=time_values, max_imf=-1)
    nIMFs = eIMFs.shape[0]
    nElements = eIMFs.shape[1]
    print('number of IMFs:', nIMFs)

    # plt.figure(figsize=(12, 9))
    # plt.subplot(nIMFs+2, 1, 1)
    # plt.plot(time_values, time_series_values, 'r')
    # plt.ylabel("Original")

    corr_data = []
    for n in range(nIMFs):
        # plt.subplot(nIMFs+2, 1, n+2)
        # plt.plot(time_values, eIMFs[n], 'g')
        # plt.ylabel("eIMF %i" %(n+1))
        #
        # # reduce number of ticks to 5
        # plt.locator_params(axis='y', nbins=5)

        distance, _ = fastdtw(time_series_values, eIMFs[n], dist=euclidean)
        corr, pval = pearsonr(time_series_values, eIMFs[n])

        corr_data.append(corr)
        print("imf", n, '-euclidean distance to original series:', distance, "; corr:", corr, ";p-value:", pval)

        np.savetxt(folder + "/imfs/IMF-" + str(n) + ".csv", eIMFs[n], delimiter=",")

    # plt.subplot(nIMFs+2, 1, nIMFs+2)
    # plt.plot(t, reconstructed_points, 'r')
    # plt.ylabel("Reconstructed")

    # plt.xlabel("Time [s]")
    # plt.tight_layout()
    # # # plt.savefig('output/eemd_example', dpi=120)
    # plt.show()

    # from sklearn.cluster import KMeans
    # import numpy as np
    #
    # ## reshape for data with single feature
    # x = np.array(corr_data).reshape(-1, 1)
    # ## clustering
    # km = KMeans(n_clusters=2, init='random', max_iter=100, n_init=1)
    # km.fit(x)
    #
    # labels = km.predict(x)
    # print("labels:", labels)
    # in row_dict we store actual meanings of rows, in my case it's russian words
