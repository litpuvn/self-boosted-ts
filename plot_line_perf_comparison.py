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
        'RNN-GRU': [131.02334625983568, 266.4695441675886, 398.88785200608567, 435.4095048978169, 454.6632587708118],

        'D-CNN': [109.44010834832092, 206.5636915277131, 289.75570778428374, 283.4357061541986, 287.71427396919216],

        'Seq2seq': [108.98431461787577, 191.61387997977397, 263.9694248835031, 281.30287590431044, 294.9900816084042],
        'Self-boosted': [80.7800216098457, 125.884031056754, 166.07100229556073, 187.81365138799708, 178.61930042880613]
    },

    'temperature': {
        'RNN-GRU': [126.27993660120123, 103.44295836003958, 96.13352645214277, 96.33634984390224, 102.19951055894728,
                        92.28384003569212, 93.03786696703503, 92.59590053266247, 93.93262426378193, 92.61991129328025,
                        91.55383227856858, 85.96056269611333],

        'D-CNN': [2.2497826994122776, 1.8967160863507884, 1.7185949667947846, 1.3502297782362, 1.3672734843484213,
                        1.69977688190889, 1.9854489919369387, 3.0239465816163285, 3.518981975370067, 4.86662707737258,
                        3.471324563701506],

        'Seq2seq': [0.007529017005649577, 0.007328292588766152, 0.006986319846778477, 0.007398312227230334,
                          0.010132788473359567, 0.00919169121037186, 0.008189049546508244, 0.00816905158546414,
                          0.009940578222472672, 0.006962016885383331]
    },

    'exchange_rate': {
        'RNN-GRU': [126.27993660120123, 103.44295836003958, 96.13352645214277, 96.33634984390224, 102.19951055894728,
                    92.28384003569212, 93.03786696703503, 92.59590053266247, 93.93262426378193, 92.61991129328025,
                    91.55383227856858, 85.96056269611333],

        'D-CNN': [2.2497826994122776, 1.8967160863507884, 1.7185949667947846, 1.3502297782362, 1.3672734843484213,
                  1.69977688190889, 1.9854489919369387, 3.0239465816163285, 3.518981975370067, 4.86662707737258,
                  3.471324563701506],

        'Seq2seq': [0.007529017005649577, 0.007328292588766152, 0.006986319846778477, 0.007398312227230334,
                    0.010132788473359567, 0.00919169121037186, 0.008189049546508244, 0.00816905158546414,
                    0.009940578222472672, 0.006962016885383331]
    },


}

dataset = 'electricity'

def create_scaler():
    global dataset
    global data

    my_data = data[dataset]

    all_performances = []
    for method, performances in my_data.items():
        all_performances = all_performances + performances

    min_v = min(all_performances)
    max_v = max(all_performances)
    scaler = MinMaxScaler(feature_range=(floor(min_v), ceil(max_v)))
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
    spl = make_interp_spline(Points, plot_data, k=3)  # type: BSpline
    xnew = np.linspace(min(Points), max(Points), 100)
    power_smooth = spl(xnew)

    return xnew, power_smooth


scaler = create_scaler()
xnew, power_smooth_ex = create_data_line('RNN-GRU', scaler=scaler)
_, power_smooth_e = create_data_line('D-CNN', scaler=scaler)
_, power_smooth_t = create_data_line('Seq2seq', scaler=scaler)
_, power_smooth_s = create_data_line('Self-boosted', scaler=scaler)

# plt.plot(plot_data, linestyle='-', marker='o', color='#8ebad9')


plt.plot(xnew, power_smooth_e, linestyle='-', color='purple')
plt.plot(xnew, power_smooth_t, linestyle='-', color='green')
plt.plot(xnew, power_smooth_ex, linestyle='-', color='blue')
plt.plot(xnew, power_smooth_s, linestyle='-', color='red')

# plt.xticks(np.arange(1, 6, step=1))
plt.xticks(np.arange(1, 6, step=1), ['t+1', 't+3', 't+5', 't+7', 't+9'])
# plt.plot(xnew, power_smooth, linestyle='-', marker='o', color='#8ebad9')

# plt.legend(['Using weather'], loc='upper right')
ax.set_ylabel('RMSE')
ax.set_xlabel('Horizon')
ax.legend(['RNN-GRU', 'Dilated CNN', 'Seq2seq', 'Self-boosted'], loc='upper left')

# ax.set_xticklabels(['Aug 23', 'Aug 24', 'Aug 25', 'day4', 'day5', 'day6', 'day7', 'day8', 'day9', 'day10', 'day11'])
# ax.set_title('Hourly average need prediction accuracy on each day')

# plt.ylim([min(plot_data) - 0.01, max(plot_data) + 0.01])

plt.savefig('results/line_perf_' + dataset + '.png')
plt.show()