import matplotlib.pyplot as plt
from math import floor
from math import ceil
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing.data import MinMaxScaler


plt.rcParams.update({'font.size': 14})

data_source = 'electricity'
# data_source = 'temperature'
# data_source = 'exchange_rate'

data = {
    'electricity': {
        'mtl': [310.8932254640111, 153.2694732475049, 110.87312435947634, 98.50992818417272],
        'mtv': [271.5210653626921, 119.99268156150028, 136.90645803755092, 165.78711422930493],
        'combined': [263.5244842602618, 79.27602621099474, 80.7800216098457, 77.52823802077538]
    },

    'temperature': {
        'mtl': [25.079681269750147, 27.934958997639935, 13.7014477435724, 5.9935892719657575],
        'mtv': [3.64280119067926, 2.4331045241278315, 3.119408830876598, 4.806783774913677],
        'combined': [2.136287365612908, 1.4173563860947447, 1.3908570276365968, 1.543066004147975]
    },
    'exchange_rate': {
        'mtl': [0.08723881198297745,  0.09590328801956823, 0.04629440341188106, 0.03760764080766279],
        'mtv': [0.012535593013500521, 0.01229572149528272, 0.01091636415044525, 0.01056433494115783],
        'combined': [0.007173354671521166, 0.006962016885383331, 0.00657861440027404, 0.007990577488602804]
    }
}

def create_scaler():
    global data_source
    global data

    my_data = data[data_source]

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


def create_data_line(data_method, scaler):

    global data_source
    global data
    data_line = data[data_source][data_method]
    plot_data = []

    for e in data_line:
        # normalized = e / total_val
        # plot_data.append(normalized)
        plot_data.append(e)

    plot_data = scaler.transform(np.array(plot_data).reshape(-1, 1))

    return plot_data.ravel()

scaler = create_scaler()
mtl = create_data_line('mtl', scaler)
mtv = create_data_line('mtv', scaler)
proposed_method = create_data_line('combined', scaler)


patterns = [ "|" , "\\" , "/" , "+" , "-", ".", "*","x", "o", "O" ]

labels = ['Lag=1', 'Lag=3', 'Lag=6', 'Lag=12']

# Setting the positions and width for the bars
pos = list(range(len(mtl)))
width = 0.2  # the width of a bar

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 6))

bar1 = plt.bar(pos, mtl, width,
               alpha=0.5,
               # color='w',
               # hatch=patterns[7],
               # edgecolor='black',
               label=labels[0])

plt.bar([p + width for p in pos], mtv, width,
        alpha=0.5,
        # color='w',
        # hatch=patterns[6],
        # edgecolor='black',
        label=labels[1])


plt.bar([p + width * 2 for p in pos], proposed_method, width,
        alpha=0.5,
        # color='w',
        # hatch=patterns[8],
        # edgecolor='black',
        label=labels[3])


# Setting axis labels and ticks
ax.set_ylabel('RMSE')
# ax.set_xlabel('City')
# ax.set_title('Comparison of normalized RMSE with respect to lag time')
ax.set_xticks([p + 1 * width for p in pos])
ax.set_xticklabels(labels)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 5)
# plt.ylim([0, 1])

# Adding the legend and showing the plot
plt.legend(['   ', '   ', '                  '], loc='upper right', ncol=3)
# plt.grid()

plt.savefig('results/mtl_mtv_all_' + data_source + '.png')
plt.show()