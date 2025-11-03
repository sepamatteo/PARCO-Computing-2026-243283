import numpy as np
import matplotlib.pyplot as plt
import argparse

## Argument parser
parser = argparse.ArgumentParser(description='Process benchmark data.')
parser.add_argument('--show-plot', action='store_true', help='Display the plot')
parser.add_argument('--coo', action='store_true', help='Show COO data')
parser.add_argument('--csr', action='store_true', help='Show sequential CSR data')
parser.add_argument('--par-csr', action='store_true', help='Show parallel CSR data')
args = parser.parse_args()

show_all = not(args.coo or args.csr or args.par_csr)

# Function to read benchmarks times
def readTimes(filename):
    with open (filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# Load data
coo_times = None
csr_times = None
parallel_csr_times = None
coo_avg = None
csr_avg = None
parallel_csr_avg = None
coo_90 = None
csr_90 = None
parallel_csr_90 = None

# Compute statistics
if args.coo or show_all:
    coo_times = readTimes("COO_exec_times.txt")
    coo_avg = np.mean(coo_times)
    coo_90 = np.percentile(coo_times, 90)
    print(f"COO average: {coo_avg:.8f} ms")
    print(f"COO 90th percentile: {coo_90:.8f} ms")

if args.csr or show_all:
    csr_times = readTimes("CSR_exec_times.txt")
    csr_avg = np.mean(csr_times)
    csr_90 = np.percentile(csr_times, 90)
    print(f"CSR average: {csr_avg:.8f} ms")
    print(f"CSR 90th percentile: {csr_90:.8f} ms")

if args.par_csr or show_all:
    parallel_csr_times = readTimes("Parallel_CSR_exec_times.txt")
    parallel_csr_avg = np.mean(parallel_csr_times)
    parallel_csr_90 = np.percentile(parallel_csr_times, 90)
    print(f"Parallel CSR average: {parallel_csr_avg:.8f} ms")
    print(f"Parallel CSR 90th percentile: {parallel_csr_90:.8f} ms")

# Create plot
if coo_times is not None:
    plt.plot(coo_times, 'ro-', label='COO times')
    plt.axhline(coo_avg, color='red', linestyle='--', label=f'COO avg ({coo_avg:.5f} ms)')
    plt.axhline(coo_90, color='purple', linestyle='-.', label=f'COO 90% ({coo_90:.5f} ms)')

if csr_times is not None:
    plt.plot(csr_times, 'bo-', label='CSR times')
    plt.axhline(csr_avg, color='blue', linestyle='--', label=f'CSR avg ({csr_avg:.5f} ms)')
    plt.axhline(csr_90, color='gold', linestyle='-.', label=f'CSR 90% ({csr_90:.5f} ms)')

if parallel_csr_times is not None:
    plt.plot(parallel_csr_times, 'go-', label='Parallel CSR times')
    plt.axhline(parallel_csr_avg, color='green', linestyle='--', label=f'Parallel CSR avg ({parallel_csr_avg:.5f} ms)')
    plt.axhline(parallel_csr_90, color='yellow', linestyle='-.', label=f'Parallel CSR 90% ({parallel_csr_90:.5f} ms)')

plt.title('Benchmark: COO vs CSR execution times')
plt.xlabel('Run #')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('benchmark_plot.png')

# Checks for argument in order to show plot
if args.show_plot:
    plt.show()