import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 14})

rmse_elec = [262.1859, 208.3035, 129.2968, 141.2073, 91.1300]
rmse_temperature = [6.9762, 1.8026, 4.7262, 1.9977, 1.5910]
rmse_exchange_rate = [0.0130, 0.0097, 0.0106, 0.0153, 0.0082]

def do_sub_plot(subplot, rmse_data, width=0.2):
    for i in range(len(rmse_data)):
        rmse_v = rmse_data[i]
        subplot.bar(0.2 + i * width, rmse_v, width,
                alpha=0.5,
                # color='w',
                # hatch=patterns[6],
                # edgecolor='black',
        )

width = 0.2
fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1)
fig.suptitle('Sharing x per column, y per row')

do_sub_plot(subplot=ax1, rmse_data=rmse_elec, width=width)
do_sub_plot(subplot=ax2, rmse_data=rmse_temperature, width=width)
do_sub_plot(subplot=ax3, rmse_data=rmse_exchange_rate, width=width)

# ax1.set_xlim(0, 3)
ax1.set_ylim(80, 270)
# ax1.set_yticks([80, 150, 220])

# ax2.set_xlim(0, 3)
# ax3.set_xlim(0, 3)

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

# ax3.set(xlabel='x-label', ylabel='y-label')
fig.set_figheight(22)
fig.set_figwidth(12)

ax1.legend(['MTL', 'MTV', 'Proposed', "Self-boosted", "proposed2"], loc='upper right', ncol=5)

plt.savefig("results/avg_rmse_comparison.png")
plt.show()