"""Compare banks' expense-rate pass-through during hikes and cuts."""
from pathlib import Path

import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"
for path in (PROC, OUT_TAB, OUT_FIG):
    path.mkdir(parents=True, exist_ok=True)

rows = []
def summarize(label, estimates):
    d = estimates.dropna(subset=["beta_Exp_up", "beta_Exp_down"]).copy()
    diff = d["beta_Exp_up"] - d["beta_Exp_down"]
    paired = stats.ttest_rel(d["beta_Exp_up"], d["beta_Exp_down"])
    return {
        "Sample": label,
        "N banks": int(len(d)),
        "Mean beta_Exp_up": d["beta_Exp_up"].mean(),
        "Mean beta_Exp_down": d["beta_Exp_down"].mean(),
        "Mean difference (up - down)": diff.mean(),
        "Median difference (up - down)": diff.median(),
        "Paired t-stat": paired.statistic,
        "Paired p-value": paired.pvalue,
        "Share with diff > 0": (diff > 0).mean(),
    }

full = pd.read_csv(PROC / "bank_betas_asym.csv")
pre07 = pd.read_csv(PROC / "bank_betas_asym_pre07.csv")
rows.append(summarize("Full sample (1984-2017)", full))

# Divide the full-sample asymmetric estimation sample into bank-size terciles
# using the time-series mean of total assets, the same measure as in Script 03.
size = pd.read_csv(PROC / "bank_betas.csv")[["idrssd", "mean_asset"]]
full_sized = full.merge(size, on="idrssd", how="left")
full_sized = full_sized.dropna(subset=["beta_Exp_up", "beta_Exp_down", "mean_asset"])
full_sized["Size tercile"] = pd.qcut(
    full_sized["mean_asset"], 3, labels=["Small", "Medium", "Large"]
)
for group in ["Small", "Medium", "Large"]:
    rows.append(summarize(group, full_sized[full_sized["Size tercile"] == group]))
rows.append(summarize("Pre-2007 (re-estimated)", pre07))

summary = pd.DataFrame(rows).round(4)
summary.to_csv(OUT_TAB / "table5_asymmetry.csv", index=False)
print(summary.to_string(index=False))

# Figure 8 compares the two cumulative betas directly; the 45-degree line
# represents equal pass-through. Plot limits use the central 99% of values so
# a handful of noisy estimates do not make the main pattern unreadable.
d = full.dropna(subset=["beta_Exp_up", "beta_Exp_down"])
low = min(d["beta_Exp_up"].quantile(0.005), d["beta_Exp_down"].quantile(0.005), 0)
high = max(d["beta_Exp_up"].quantile(0.995), d["beta_Exp_down"].quantile(0.995))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.scatter(d["beta_Exp_down"], d["beta_Exp_up"], s=5, alpha=0.16,
           color="steelblue", rasterized=True)
ax.plot([low, high], [low, high], color="crimson", lw=1.5, ls="--",
        label="45-degree line (symmetric)")
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_xlabel(r"$\beta^{Exp}_{down}$")
ax.set_ylabel(r"$\beta^{Exp}_{up}$")
ax.set_title("Expense-rate beta on rate hikes vs. rate cuts")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig8_asym_scatter.png", bbox_inches="tight")
plt.close(fig)

# Figure 9 shows the full-sample distribution of bank-level hike-minus-cut
# differences. The vertical line marks equal pass-through.
diff = d["beta_Exp_up"] - d["beta_Exp_down"]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.hist(diff, bins=60, color="steelblue", alpha=0.82, edgecolor="white")
ax.axvline(0, color="crimson", lw=1.5, ls="--", label="Equal pass-through")
ax.axvline(diff.mean(), color="black", lw=1.2, ls=":", label=f"Mean = {diff.mean():.3f}")
ax.set_xlabel(r"$\beta^{Exp}_{up} - \beta^{Exp}_{down}$")
ax.set_ylabel("Banks")
ax.set_title("Distribution of bank-level pass-through differences")
ax.set_xlim(diff.quantile(0.005), diff.quantile(0.995))
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig9_asym_diff.png", bbox_inches="tight")
plt.close(fig)
