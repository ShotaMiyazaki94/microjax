import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", font_scale=1.2, font="serif")

df = pd.read_csv("paper/speed_comparison_uniform.csv")

fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True, sharex=True, )
colors = ['r', 'g', 'b', 'c', 'm']

rho_labels = [r"$\rho=0.1$", r"$\rho=0.01$", r"$\rho=10^{-3}$"]
for i, (rho, group) in enumerate(df.groupby("rho")):
    group = group.sort_values("npts")
    ax[0].plot(group["npts"], group["time_mj_mean"],
             '*', label=f"microjax, "+rho_labels[i], color=colors[i % len(colors)], ms=10)
    ax[0].plot(group["npts"], group["time_vbb_mean"],
             'o', label=f"VBBinaryLensing, "+rho_labels[i], color=colors[i % len(colors)], alpha=0.6, ms=10)
df = pd.read_csv("paper/speed_comparison_limbdark.csv")
for i, (rho, group) in enumerate(df.groupby("rho")):
    group = group.sort_values("npts")
    ax[1].plot(group["npts"], group["time_mj_mean"],
             '*', label=f"microjax, "+rho_labels[i], color=colors[i % len(colors)], ms=10)
    ax[1].plot(group["npts"], group["time_vbb_mean"],
             'o', label=f"VBBinaryLensing, "+rho_labels[i], color=colors[i % len(colors)], alpha=0.6, ms=10)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Number of points (npts)")
ax[1].set_xlabel("Number of points (npts)")
ax[0].set_xticks([10, 30, 100, 300], [str(x) for x in [10, 30, 100, 300]])
ax[0].set_yticks([0.01, 0.1, 1, 10], [str(y) for y in [0.01, 0.1, 1, 10]])
ax[0].set_ylabel("Execution time [s]")
ax[0].set_title("uniform brightness source")
ax[1].set_title("limb-darkening source")
ax[0].legend(loc="upper left", fontsize=12)
#ax[1].legend(loc="lower right", fontsize=10)
ax[0].grid(True, which='both', ls='--', alpha=0.5)
ax[1].grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("paper/figure/timing_vs_npts.pdf", dpi=300)
plt.close()