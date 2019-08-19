from PyEMD import EEMD
from PyEMD import CEEMDAN
# import matplotlib as plt
import pylab as plt

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


if __name__ == "__main__":
    # s = np.random.random(100)

    # ## CEEMD
    # s = np.random.random(100)
    # ceemdan = CEEMDAN()
    # cIMFs = ceemdan(s)

    np.random.seed(0)

    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    S = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    S += 5 * sin(11, 2.7)
    S += 3 * sin(14, 1.6)
    S += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    S += t ** 2.1 - t

    # Execute EEMD on S
    eemd = CEEMDAN()
    # eIMFs = eemd(s)
    eIMFs = eemd.ceemdan(S, t)
    nIMFs = eIMFs.shape[0]

    print('number of IMFs:', nIMFs)
    # Plot results
    plt.figure(figsize=(12,9))
    plt.subplot(nIMFs+2, 1, 1)
    plt.plot(t, S, 'r')
    plt.ylabel("Original")

    reconstructed_points = np.sum(eIMFs, axis=0)

    for n in range(nIMFs):
        plt.subplot(nIMFs+2, 1, n+2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)

        distance, _ = fastdtw(S, eIMFs[n], dist=euclidean)

        print('euclidean distance to original series:', distance)

    plt.subplot(nIMFs+2, 1, nIMFs+2)
    plt.plot(t, reconstructed_points, 'r')
    plt.ylabel("Reconstructed")

    plt.xlabel("Time [s]")
    plt.tight_layout()
    # plt.savefig('output/eemd_example', dpi=120)
    plt.show()