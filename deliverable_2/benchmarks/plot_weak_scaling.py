import matplotlib.pyplot as plt
import numpy as np

# ─── Data ────────────────────────────────────────────────
p = np.array([1, 2, 4, 8, 16])

global_rows   = np.array([10_000, 20_000, 40_000, 80_000, 160_000])
nnz_per_spmv  = np.array([1e6, 4e6, 16e6, 64e6, 256e6])          # grows linearly with P

time_ms       = np.array([1.517, 2.410, 3.461, 3.957, 132.7])
gflops        = np.array([1.31, 8.96, 3.96, 3.96, 3.84])
ghosts        = np.array([0, 10_000, 10_000, 70_000, 150_000])
comm_fraction = np.array([3.7, 1.1, 10.1, 62.4, 58.7])           # in %

# ─── Plotting ────────────────────────────────────────────
plt.style.use('ggplot')
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Weak Scaling Behavior – SpMV on Growing Problem Size", fontsize=16, y=0.98)

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
ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

# ── 2. Communication fraction ───────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("Communication Fraction (%)")
ax2.plot(p, comm_fraction, 's-', color='darkorange', lw=2.5, ms=9)
ax2.set_ylabel('Communication fraction (%)')
ax2.set_ylim(0, max(comm_fraction)*1.15)
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

# ── 4. Problem size indicators ──────────────────────────
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("Problem Size Scaling (should be linear in weak scaling)")
ax4.plot(p, global_rows, 'D-', color='purple', lw=2.5, ms=9, label='Global rows')
ax4.plot(p, nnz_per_spmv, 'o--', color='slateblue', lw=2.2, ms=8, label='nnz per SpMV')
ax4.set_yscale('log', base=10)
ax4.set_ylabel('Size (log scale)')
ax4.set_xlabel('Number of processes P')
ax4.set_xscale('log', base=2)
ax4.set_xticks(p)
ax4.set_xticklabels([f"P={int(x)}" for x in p])
ax4.grid(True, alpha=0.35)
ax4.legend(loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../docs/report/weak_scaling.pdf', bbox_inches='tight', dpi=300)