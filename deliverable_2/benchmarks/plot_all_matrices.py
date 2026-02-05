import matplotlib.pyplot as plt
import numpy as np

# Define processes and matrices
processes = [1, 2, 4, 8, 16, 32]
matrices = ["1138_bus", "bcsstk18", "cage14", "nlpkkt160", "Queen_4147"]

# Data dictionary (extracted from your tables; adjust if transcription errors)
data = {
    "1138_bus": {
        "avg_time": [0.021, 0.096, 0.025, 16.694, 26.521, 41.947],
        "speedup": [1.000, 0.219, 0.840, 0.001, 0.001, 0.001],
        "efficiency": [100.000, 10.938, 21.000, 0.016, 0.005, 0.002],
        "comm_pct": [0.444, 71.273, 78.378, 79.849, 83.159, 78.440],
        "gflops": [0.240, 0.054, 0.246, 0.003, 0.001, 0.001],
        "avg_ghost": [0, 289, 245, 150, 83, 44],
    },
    "bcsstk18": {
        "avg_time": [0.254, 0.364, 0.227, 13.270, 28.708, 47.403],
        "speedup": [1.000, 0.698, 1.119, 0.019, 0.009, 0.005],
        "efficiency": [100.000, 34.890, 27.974, 0.239, 0.055, 0.017],
        "comm_pct": [0.065, 11.785, 81.004, 62.118, 69.789, 47.258],
        "gflops": [0.633, 0.442, 0.796, 0.012, 0.005, 0.003],
        "avg_ghost": [0, 4884, 6149, 5191, 3386, 2784],
    },
    "cage14": {
        "avg_time": [54.803, 38.853, 33.153, 35.587, 40.104, 36.541],
        "speedup": [1.000, 1.411, 1.653, 1.540, 1.367, 1.500],
        "efficiency": [100.000, 70.526, 41.326, 19.250, 8.541, 4.687],
        "comm_pct": [0.002, 18.556, 62.885, 74.660, 85.494, 34.529],
        "gflops": [0.990, 1.396, 1.636, 0.981, 1.061, 1.039],
        "avg_ghost": [0, 752034, 934533, 838105, 627365, 425588],
    },
    "nlpkkt160": {
        "avg_time": [227.970, 147.598, 49.676, 45.043, 60.561, 54.878],
        "speedup": [1.000, 1.545, 4.589, 5.061, 3.764, 4.154],
        "efficiency": [100.000, 77.227, 114.728, 63.265, 23.527, 12.982],
        "comm_pct": [0.001, 20.187, 60.805, 63.956, 72.382, 45.714],
        "gflops": [1.045, 1.611, 4.788, 5.280, 3.927, 3.532],
        "avg_ghost": [0, 2099200, 2098880, 1305440, 716720, 374360],
    },
    "Queen_4147": {
        "avg_time": [295.379, 198.377, 82.743, 76.940, 171.558, 340.880],
        "speedup": [1.000, 1.489, 3.570, 3.839, 1.722, 0.867],
        "efficiency": [100.000, 74.449, 89.246, 47.989, 10.761, 2.708],
        "comm_pct": [0.000, 14.761, 62.947, 79.612, 91.231, 88.495],
        "gflops": [1.127, 1.673, 4.032, 4.336, 1.577, 0.978],
        "avg_ghost": [0, 2073552, 3110312, 3626284, 3764212, 3135765],
    },
}

# Metrics to plot
metrics = ["avg_time", "speedup", "efficiency", "comm_pct", "gflops", "avg_ghost"]
titles = [
    "Average Time (s)",
    "Speedup",
    "Efficiency (%)",
    "Communication %",
    "GFLOPs",
    "Average Ghost Count",
]
y_labels = ["Time (s)", "Speedup", "Efficiency (%)", "Comm %", "GFLOPs", "Ghost Count"]

# Create figure with subplots (3 rows, 2 columns)
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    ax = axs[i]
    for matrix in matrices:
        ax.plot(processes, data[matrix][metric], marker="o", label=matrix)
    ax.set_xscale("log", base=2)
    ax.set_xticks(processes)
    ax.set_xticklabels(processes)
    ax.set_xlabel("Number of Processes")
    ax.set_ylabel(y_labels[i])
    ax.set_title(titles[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("../docs/report/all_matrices.pdf")
plt.show()
