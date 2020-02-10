# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data_dir = 'data'
# filename = 'hourly_clean_electricity.csv'
# filename = 'temperature.csv'
filename = 'time_exchange_rate.csv'
df = pd.read_csv(os.path.join(data_dir, filename), sep=',', header=0, index_col=0, parse_dates=True)
if 'electric' in filename:
    df.rename(columns={'avg_electricity': 'load'}, inplace=True)
elif 'temperature' in filename:
    df.rename(columns={'Temperature': 'load'}, inplace=True)

else:
    df.rename(columns={'rate': 'load'}, inplace=True)

# multiple line plot
# plt.plot(df['x'], df['y1'], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
# plt.plot(df['x'], df['y2'], marker='', color='olive', linewidth=2)
plt.plot(df.index, df['load'], marker='', color='red', linewidth=2, linestyle='dashed', label="toto")
# plt.legend()

# df.plot()

plt.show()