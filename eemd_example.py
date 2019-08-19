from PyEMD import EEMD
from PyEMD import CEEMDAN
# import matplotlib as plt
import pylab as plt

import numpy as np

if __name__ == "__main__":
    # s = np.random.random(100)

    # ## CEEMD
    # s = np.random.random(100)
    # ceemdan = CEEMDAN()
    # cIMFs = ceemdan(s)

    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    S = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    S += 5 * sin(11, 2.7)
    S += 3 * sin(14, 1.6)
    S += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    S += t ** 2.1 - t

    # Execute EEMD on S
    eemd = EEMD()
    # eIMFs = eemd(s)
    eIMFs = eemd.eemd(S, t)
    nIMFs = eIMFs.shape[0]


    # Plot results
    plt.figure(figsize=(12,9))
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(t, S, 'r')

    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    # plt.savefig('output/eemd_example', dpi=120)
    plt.show()