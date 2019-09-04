import seaborn as sns
from sklearn import preprocessing
import numpy as np

sns.set()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14,
                    'legend.fontsize': 14,
                     })

fig, ax = plt.subplots(figsize=(6, 2))


data_source = 'exchange_rate'

data = {
    'electricity': [126.27993660120123, 103.44295836003958, 96.13352645214277, 96.33634984390224, 102.19951055894728,
                    92.28384003569212, 93.03786696703503, 92.59590053266247, 93.93262426378193, 92.61991129328025,
                    91.55383227856858, 85.96056269611333],

    'temperature': [2.2497826994122776, 1.8967160863507884, 1.7185949667947846, 1.3502297782362, 1.3672734843484213,
                    1.69977688190889, 1.9854489919369387, 3.0239465816163285, 3.518981975370067, 4.86662707737258,
                    3.471324563701506],

    'exchange_rate': [0.007529017005649577, 0.007328292588766152, 0.006986319846778477, 0.007398312227230334,
                      0.010132788473359567, 0.00919169121037186, 0.008189049546508244, 0.00816905158546414,
                      0.009940578222472672, 0.006962016885383331]

}

data_line = data[data_source]

min_val = min(data_line)
max_val = max(data_line)
total_val = sum(data_line)
# scaler = preprocessing.StandardScaler()
#
# data_line = np.array(data_line)
# plot_tmp = scaler.fit_transform(data_line.reshape((-1,1)))
# plot_data = plot_tmp

plot_data = []
total = sum(data_line)
for e in data_line:
    normalized = e / total_val
    plot_data.append(normalized)


plt.plot(plot_data, linestyle='-', marker='o', color='#8ebad9')

# plt.legend(['Using weather'], loc='upper right')
ax.set_ylabel('RMSE Coefficient')
ax.set_xlabel('IMF')
# ax.set_xticklabels(['Aug 23', 'Aug 24', 'Aug 25', 'day4', 'day5', 'day6', 'day7', 'day8', 'day9', 'day10', 'day11'])
# ax.set_title('Hourly average need prediction accuracy on each day')

# plt.ylim([min(plot_data) - 0.01, max(plot_data) + 0.01])

plt.savefig('results/line_' + data_source + '.png')
plt.show()