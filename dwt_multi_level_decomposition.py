from pywt import wavedec
import numpy as np
from scipy import signal
import pywt
(cA, cD) = pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')

print('approximation:', cA)
print('details:', cD)

# coeffs = wavedec([1, 2, 3, 4, 5, 6, 7, 8], 'db1')
#
# print(coeffs)

def DFT(x):
    """
    Compute the discrete Fourier Transform of the 1D array x
    :param x: (array)
    """

    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

t = np.linspace(0, 500, 500)
s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

fft = np.fft.fft(s)


f, ti, z = signal.stft(s)

print('size: ', len(fft))
for i in range(2):
    print("Value at index {}:\t{}".format(i, fft[i + 1]), "\nValue at index {}:\t{}".format(fft.size -1 - i, fft[-1 - i]))

#
# dft = DFT(s)
# for i in range(2):
#     print("Value at DFT index {}:\t{}".format(i, dft[i + 1]), "\nValue at index {}:\t{}".format(dft.size -1 - i, dft[-1 - i]))