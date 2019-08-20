from PyEMD import EEMD
from PyEMD import CEEMDAN
# import matplotlib as plt
import pylab as plt

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


if __name__ == "__main__":

    filepath = 'data/data-2011.csv'
    time_series_values = np.loadtxt(filepath, delimiter=',', usecols=1, skiprows=1)

    time_values = np.linspace(0, 1, len(time_series_values))

    # reconstructed_points = np.sum(eIMFs, axis=0)

    # Execute EEMD on S
    eemd = CEEMDAN(trials=10, epsilon=0.005, ext_EMD=None)
    eIMFs = eemd.ceemdan(S=time_series_values, T=time_values, max_imf=-1)
    nIMFs = eIMFs.shape[0]

    print('number of IMFs:', nIMFs)

    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs+2, 1, 1)
    plt.plot(time_values, time_series_values, 'r')
    plt.ylabel("Original")

    for n in range(nIMFs):
        plt.subplot(nIMFs+2, 1, n+2)
        plt.plot(time_values, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))

        # reduce number of ticks to 5
        plt.locator_params(axis='y', nbins=5)

        distance, _ = fastdtw(time_series_values, eIMFs[n], dist=euclidean)

        print("imf", n, '-euclidean distance to original series:', distance)

    # plt.subplot(nIMFs+2, 1, nIMFs+2)
    # plt.plot(t, reconstructed_points, 'r')
    # plt.ylabel("Reconstructed")

    plt.xlabel("Time [s]")
    plt.tight_layout()
    # # plt.savefig('output/eemd_example', dpi=120)
    plt.show()