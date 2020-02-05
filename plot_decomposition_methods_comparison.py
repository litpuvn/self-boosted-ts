import seaborn as sns
from sklearn import preprocessing
import numpy as np
from math import floor
from math import ceil

from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing.data import MinMaxScaler

sns.set()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14,
                    'legend.fontsize': 14,
                     })

fig, ax = plt.subplots(figsize=(8, 5))


data = {
    'electricity': {
        'additive': [93.94702670080696, 124.74408157952728, 130.0006408594057, 130.9219017180751, 132.4146409585086],
        'multiplicative': [93.58474590068776, 126.09260446901177, 133.20162943699566, 131.01328137716237, 137.37682483140608],
        'Self-boosted': [80.7800216098457, 113.76110531783974, 152.59680725743004, 166.8524359051895, 172.76464795213244]
    },

    'temperature': {
        'additive': [2.3167724777072203, 2.6556840016789893, 2.9014753497865926, 3.0724093522839846, 3.140775000266926],
        'multiplicative': [2.3337837085006226, 3.122145197327413, 3.908123412309968, 4.019262838640005, 4.299082504883039],
        'Self-boosted': [1.8640028533970154, 2.9676957788976264 , 3.8451002506918583, 4.03406402512719, 5.586341553019636]
    },

    'exchange_rate': {
        'additive': [0.011245275093378061, 0.012224529530525502, 0.013034005971612313, 0.014841941314902223, 0.019432741276142734],
        'multiplicative': [0.01348146513976495, 0.01623901308995848, 0.01644604990722094, 0.020293536414275674, 0.02025980730150017],
        'Self-boosted': [0.0068517864378336004, 0.008769284833771972, 0.00907317391521684, 0.010189584351223335, 0.010705994501009104]

    },


}

# dataset = 'electricity'
# dataset = 'temperature'
dataset = 'exchange_rate'

def create_scaler():
    global dataset
    global data

    my_data = data[dataset]

    all_performances = []
    for method, performances in my_data.items():
        all_performances = all_performances + performances

    min_v = min(all_performances)
    max_v = max(all_performances)
    lower = floor(min_v)
    upper = ceil(max_v)

    if max_v < 1:
        upper = max_v
        lower = min_v

    scaler = MinMaxScaler(feature_range=(lower, upper))
    scaler.fit(np.array(all_performances).reshape(-1, 1))

    return scaler




def create_data_line(data_source, scaler):

    global dataset
    global data
    data_line = data[dataset][data_source]
    total_val = sum(data_line)
    plot_data = []
    Points = []
    index = 1
    for e in data_line:
        # normalized = e / total_val
        # plot_data.append(normalized)
        plot_data.append(e)
        Points += [index]
        index = index + 1

    plot_data = scaler.transform(np.array(plot_data).reshape(-1, 1))
    spl = make_interp_spline(Points, plot_data, k=1)  # type: BSpline
    xnew = np.linspace(min(Points), max(Points), 9)
    power_smooth = spl(xnew)

    return xnew, power_smooth


scaler = create_scaler()
xnew, power_smooth_additive = create_data_line('additive', scaler=scaler)
_, power_smooth_multiplicative = create_data_line('multiplicative', scaler=scaler)
_, power_smooth_s = create_data_line('Self-boosted', scaler=scaler)

# plt.plot(plot_data, linestyle='-', marker='o', color='#8ebad9')


plt.plot(xnew, power_smooth_additive, linestyle='-', marker='o', color='purple')
plt.plot(xnew, power_smooth_multiplicative, linestyle='-', marker='*', color='green')
plt.plot(xnew, power_smooth_s, linestyle='-', marker='D', color='red')

plt.xticks(np.arange(1, 6, step=1), ['t+1', 't+3', 't+5', 't+7', 't+9'])
# plt.plot(xnew, power_smooth, linestyle='-', marker='o', color='#8ebad9')

# plt.legend(['Using weather'], loc='upper right')
ax.set_ylabel('RMSE')
ax.set_xlabel('Horizon')
# ax.legend(['RNN-GRU', 'Dilated CNN', 'Seq2seq', 'Self-boosted'], loc='upper left')
ax.legend(  ['            ', '            ', '            '], loc='upper left')
# ax.set_facecolor('white')
# ax.set_xticklabels(['Aug 23', 'Aug 24', 'Aug 25', 'day4', 'day5', 'day6', 'day7', 'day8', 'day9', 'day10', 'day11'])
# ax.set_title('Hourly average need prediction accuracy on each day')

# plt.ylim([min(plot_data) - 0.01, max(plot_data) + 0.01])

plt.savefig('results/line_decomposition_methods_' + dataset + '.png')
plt.show()