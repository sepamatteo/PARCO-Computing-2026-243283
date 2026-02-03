import matplotlib.pyplot as plt
import numpy as np

# ─── Data ────────────────────────────────────────────────
# Updated with values from the attached table
p = np.array([1, 2, 4, 8, 16, 32])

# Data extracted from table rows
time_ms       = np.array([54.803, 38.853, 33.153, 35.587, 40.104, 36.541])
speedup       = np.array([1.00, 1.410, 1.653, 1.540, 1.366, 1.499])
efficiency    = np.array([100, 70.5, 41.325, 19.25, 8.537, 4.684])
comm_fraction = np.array([0.002, 18.556, 62.885, 74.660, 85.494, 34.529])
gflops        = np.array([0.990, 1.396, 1.636, 0.981, 1.061, 1.039])
ghosts        = np.array([0, 752034, 934533, 838105, 627365, 425588])

# ─── Plotting ────────────────────────────────────────────
plt.style.use('ggplot')
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Performance Scaling Analysis – SpMV", fontsize=16, y=0.98)

# ── 1. Time per SpMV + GFLOPS ───────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("Time per SpMV & Achieved GFLOPS")

ln1 = ax1.plot(p, time_ms, 'o-', color='crimson', lw=2.4, ms=9, label='Time / SpMV (ms)')
ax1.set_ylabel('Time per SpMV (ms)', color='crimson')
ax1.tick_params(axis='y', labelcolor='crimson')
ax1.set_xscale('log', base=2)
ax1.set_xticks(p)
ax1.set_xticklabels([f"P={int(x)}" for x in p])
ax1.grid(True, alpha=0.35)

ax1b = ax1.twinx()
ln2 = ax1b.plot(p, gflops, 's--', color='teal', lw=2.1, ms=8, label='GFLOPS')
ax1b.set_ylabel('GFLOPS (avg)', color='teal')
ax1b.tick_params(axis='y', labelcolor='teal')

lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', framealpha=0.9)

# ── 2. Communication fraction ───────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Communication Fraction (%)")
ax2.plot(p, comm_fraction, 's-', color='darkorange', lw=2.5, ms=9)
ax2.set_ylabel('Communication fraction (%)')
ax2.set_ylim(-5, 100)  # Adjusted limits for 0-100%
ax2.set_xscale('log', base=2)
ax2.set_xticks(p)
ax2.set_xticklabels([f"P={int(x)}" for x in p])
ax2.grid(True, alpha=0.35)
ax2.axhline(y=50, color='grey', ls='--', alpha=0.6, label='50% threshold')
ax2.legend()

# ── 3. Ghost points per rank ────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("Ghost Points per Rank")
ax3.bar(p, ghosts, color='indianred', alpha=0.75, width=0.7*p**0.35)
ax3.set_ylabel('Ghosts (avg / rank)')
ax3.set_xlabel('Number of processes P')
ax3.set_xscale('log', base=2)
ax3.set_xticks(p)
ax3.set_xticklabels([f"P={int(x)}" for x in p])
ax3.grid(True, axis='y', alpha=0.35)

# ── 4. Speedup & Efficiency (Replaces Problem Size) ─────
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("Speedup & Efficiency")

# Plot Speedup (Left Axis)
ln3 = ax4.plot(p, speedup, 'D-', color='purple', lw=2.5, ms=9, label='Speedup')
ax4.set_ylabel('Speedup')
ax4.set_xlabel('Number of processes P')
ax4.set_xscale('log', base=2)
ax4.set_xticks(p)
ax4.set_xticklabels([f"P={int(x)}" for x in p])
ax4.grid(True, alpha=0.35)

# Plot Efficiency (Right Axis)
ax4b = ax4.twinx()
ln4 = ax4b.plot(p, efficiency, 'o--', color='slateblue', lw=2.2, ms=8, label='Efficiency (%)')
ax4b.set_ylabel('Efficiency (%)', color='slateblue')
ax4b.tick_params(axis='y', labelcolor='slateblue')
ax4b.set_ylim(0, 110)

# Combined Legend
lines2 = ln3 + ln4
labels2 = [l.get_label() for l in lines2]
ax4.legend(lines2, labels2, loc='upper right', framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../docs/report/strong_scaling.pdf', bbox_inches='tight', dpi=300)
plt.show()