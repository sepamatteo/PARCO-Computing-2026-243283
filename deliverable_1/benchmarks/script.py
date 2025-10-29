import numpy as np
import matplotlib.pyplot as plt

# function to open benchmarks files
def readTimes(filename):
    with open (filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]
        
        
coo_times = readTimes("COO_exec_times.txt")
csr_times = readTimes("CSR_exec_times.txt")
parallel_csr_times = readTimes("Parallel_CSR_exec_times.txt")

# calculates average
coo_avg = np.mean(coo_times)
csr_avg = np.mean(csr_times)
parallel_csr_avg = np.mean(parallel_csr_times)
coo_90 = np.percentile(coo_times, 90)
csr_90 = np.percentile(csr_times, 90)
parallel_csr_90 = np.percentile(parallel_csr_times, 90)

print(f"COO average: {coo_avg:.8f} ms")
print(f"CSR average: {csr_avg:.8f} ms")
print(f"Parallel CSR average: {parallel_csr_avg:.8f} ms")
print(f"COO 90th percentile: {coo_90:.8f} ms")
print(f"CSR 90th percentile: {csr_90:.8f} ms")
print(f"Parallel CSR 90th percentile: {parallel_csr_90:.8f} ms")

# plot
plt.figure(figsize=(8,5))
plt.plot(coo_times, 'ro-', label='COO times')
plt.plot(csr_times, 'bo-', label='CSR times')
plt.plot(parallel_csr_times, 'go-', label='Parallel CSR times')

plt.axhline(coo_avg, color='red', linestyle='--', label=f'COO avg ({coo_avg:.5f} ms)')
plt.axhline(csr_avg, color='blue', linestyle='--', label=f'CSR avg ({csr_avg:.5f} ms)')
plt.axhline(parallel_csr_avg, color='green', linestyle='--', label=f'Parallel CSR avg ({parallel_csr_avg:.5f} ms)')
plt.axhline(coo_90, color='purple', linestyle='-.', label=f'COO 90% percentile ({coo_90:.5f} ms)')
plt.axhline(csr_90, color='gold', linestyle='-.', label=f'CSR 90% percentile ({csr_90:.5f} ms)')
plt.axhline(parallel_csr_90, color='yellow', linestyle='-.', label=f'Parallel CSR 90th percentile ({parallel_csr_90:.5f} ms)')

plt.title('Benchmark: COO vs CSR execution times')
plt.xlabel('Run #')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('benchmark_plot.png')
#plt.show()