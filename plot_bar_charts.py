import matplotlib.pyplot as plt



plt.rcParams.update({'font.size': 14})


data_source = 'exchange_rate'
data = {
    'electricity': {
        'mtl': [310.8932254640111, 153.2694732475049, 110.87312435947634, 98.50992818417272],
        'mtv': [271.5210653626921, 119.99268156150028, 136.90645803755092, 165.78711422930493],
        'combined': [271.4936402595344, 98.47565612996547, 85.96056269611333, 88.95372389402823 ]
    },

    'temperature': {
        'mtl': [25.079681269750147, 27.934958997639935, 13.7014477435724, 5.9935892719657575],
        'mtv': [3.64280119067926, 2.4331045241278315, 3.119408830876598, 4.806783774913677],
        'combined': [2.5961495873263036, 1.4645261660987718, 1.3908570276365968, 1.917633692585684 ]
    },
    'exchange_rate': {
        'mtl': [0.08723881198297745,  0.09590328801956823, 0.04629440341188106, 0.03760764080766279],
        'mtv': [0.012535593013500521, 0.01229572149528272, 0.01091636415044525, 0.01056433494115783],
        'combined': [0.010766217687836718, 0.007472192457683756, 0.009556636679286487, 0.008185429839249353]
    }
}

data_for_source = data[data_source]

original_mtl = data_for_source['mtl']
original_mtv = data_for_source['mtv']
original_proposed_method = data_for_source['combined']

mtl = []
mtv = []
proposed_method = []

min_y = 1
max_y = 0

for lag in range(len(original_mtl)):
    mtl_at_lag = original_mtl[lag]
    mtv_at_lag = original_mtv[lag]
    combined_at_lag = original_proposed_method[lag]

    total_val = mtl_at_lag + mtv_at_lag + combined_at_lag

    mtl_norm = mtl_at_lag / total_val
    mtv_norm = mtv_at_lag / total_val
    combined_norm = combined_at_lag / total_val

    mtl.append(mtl_norm)
    mtv.append(mtv_norm)
    proposed_method.append(combined_norm)

    tmp_min = min([mtl_norm, mtv_norm, combined_norm])
    tmp_max = max([mtl_norm, mtv_norm, combined_norm])
    if min_y > tmp_min:
        min_y = tmp_min

    if max_y < tmp_max:
        max_y = tmp_max


### other patterns: https://www.w3resource.com/graphics/matplotlib/barchart/matplotlib-barchart-exercise-17.php
# Input data; groupwise

#
# mtl = [0.86, 0.89, 0.88, 0.90]
# mtv =    [0.84, 0.87, 0.85, 0.865]
# proposed_method =   [0.82, 0.83, 0.83, 0.845]

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
ax.set_ylabel('Normalized RMSE')
# ax.set_xlabel('City')
ax.set_title('Comparison of normalized RMSE with respect to lag time')
ax.set_xticks([p + 1 * width for p in pos])
ax.set_xticklabels(labels)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 5)
plt.ylim([0, 1])

# Adding the legend and showing the plot
plt.legend(['MTL', 'MTV', 'Proposed'], loc='upper right', ncol=3)
# plt.grid()

plt.savefig('results/mtl_mtv_all_' + data_source + '.png')
plt.show()