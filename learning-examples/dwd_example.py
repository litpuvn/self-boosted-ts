import pywt



cA, cD = pywt.dwt([1, 2, 3, 4], wavelet='db1')
print('cA:', cA)
print('cD', cD)
print('done')