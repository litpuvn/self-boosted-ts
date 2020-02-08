import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 14})

rmse_elec = [128.2484, 94.4698, 129.4769, 97.2255, 119.1188, 94.5988, 79.1948]
rmse_temperature = [1.8026, 2.5126, 4.7262, 1.4568, 1.9977, 1.8569, 1.4504]
rmse_exchange_rate = [0.0097, 0.0232, 0.0090, 0.0093, 0.0153, 0.0284, 0.0071]

def do_sub_plot(subplot, rmse_data, width=0.2):
    for i in range(len(rmse_data)):
        rmse_v = rmse_data[i]
        subplot.bar(0.1 + i * width, rmse_v, width,
                alpha=0.5,
                # color='w',
                # hatch=patterns[6],
                # edgecolor='black',
        )

width = 0.1
fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1)
fig.suptitle('Sharing x per column, y per row')

do_sub_plot(subplot=ax1, rmse_data=rmse_elec, width=width)
do_sub_plot(subplot=ax2, rmse_data=rmse_temperature, width=width)
do_sub_plot(subplot=ax3, rmse_data=rmse_exchange_rate, width=width)

# ax1.set_xlim(0, 3)
# ax1.set_ylim(80, 270)
# ax1.set_yticks([80, 150, 220])

# ax2.set_xlim(0, 3)
# ax3.set_xlim(0, 3)

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

# ax3.set(xlabel='x-label', ylabel='y-label')
fig.set_figheight(26)
fig.set_figwidth(6)

ax1.legend(['MTL', 'MTV', 'Proposed', "Self-boosted", "proposed2", "6", "7"], loc='upper right', ncol=7)

plt.savefig("results/avg_rmse_comparison.png")
plt.show()