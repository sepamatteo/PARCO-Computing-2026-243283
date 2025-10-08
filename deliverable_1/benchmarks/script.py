import numpy as np
import matplotlib.pyplot as plt

# function to open benchmarks files
def readTimes(filename):
    with open (filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]
        
        
coo_times = readTimes("COO_exec_times.txt")
csr_times = readTimes("CSR_exec_times.txt")

# calculates average
coo_avg = np.mean(coo_times)
csr_avg = np.mean(csr_times)

print(f"COO average: {coo_avg:.8f} ms")
print(f"CSR average: {csr_avg:.8f} ms")

# plots
plt.figure(figsize=(8,5))
plt.plot(coo_times, 'ro-', label='COO times')
plt.plot(csr_times, 'bo-', label='CSR times')
plt.axhline(coo_avg, color='red', linestyle='--', label=f'COO avg ({coo_avg:.5f} ms)')
plt.axhline(csr_avg, color='blue', linestyle='--', label=f'CSR avg ({csr_avg:.5f} ms)')
plt.title('Benchmark: COO vs CSR execution times')
plt.xlabel('Run #')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()