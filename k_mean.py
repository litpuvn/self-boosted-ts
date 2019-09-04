from sklearn.cluster import KMeans
import numpy as np


e_data = [0.03788537229453167, 0.11665254672338256, 0.6616966252625939, 0.8188354176400108, 0.12691361435242227,
0.06386014784416012, 0.07511872052127352,  0.08605094713265904, 0.0960078537374374, 0.24924330938599507, 0.33917566898688256,
0.20996469152596825, 0.20252115592293604
          ]

ex_data = [0.0018083438641824277, 0.0011816417020305075, 0.0269321735714387, 0.02935233686301139, 0.0772517772509588,
0.10128753647345597, 0.21677932228933564,  0.363447769185964, 0.5248504892948052, 0.8116422657606988, 0.5276822732467228
           ]

t_data =[0.02418369596115859, 0.06981265116477096, 0.3207234060637974,  0.34616989724911584, 0.11893893375977603,
0.15421717124665493, 0.22699158932039523, 0.218101513535665, 0.24227497459235725, 0.8318681988296313, 0.8313796140054373,
0.5302984991620289
         ]

correlation_data = {
    # 'electricity': e_data,
    'exchange_rate': ex_data,
    # 'temperature': t_data
}


my_data = ex_data
tmp = dict()

for idx, v in enumerate(my_data):
    tmp[str(v)] = "imf" + str(idx)

imf_sorted = [tmp[str(v)] for v in sorted(my_data)]

imf_sorted.reverse()

print("Total:", len(imf_sorted), imf_sorted)
#
# for name, corr_data in correlation_data.items():
#     print("Clustering imfs of data", name)
#     ## reshape for data with single feature
#     x = np.array(corr_data).reshape(-1, 1)
#     ## clustering
#     km = KMeans(n_clusters=2, init='random', max_iter=100, n_init=1)
#     km.fit(x)
#
#     labels = km.predict(x)
#     print("labels:", labels)
