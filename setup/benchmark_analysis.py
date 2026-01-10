

## Plot benchmark csv

## For each step, compute mean, and std deviation of lap times
## Plot with error bars

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
DUMP_PATH = "../trainings/sac_wall/benchmark_results.csv"
df = pd.read_csv(DUMP_PATH, index_col=0)
steps = []
means = []
stds = []
for index, row in df.iterrows():
    steps.append(int(index))
    lap_times = row.dropna().values
    means.append(np.mean(lap_times))
    stds.append(np.std(lap_times))
plt.errorbar(steps, means, yerr=stds, fmt='-o', ecolor='r', capsize=5)
plt.xlabel("Training Steps")
plt.ylabel("Lap Time (s)")
plt.title("Benchmark Results Over Training Steps")
plt.grid()
plt.savefig(os.path.join(os.path.dirname(DUMP_PATH), "benchmark_plot.png"))