"""
Bank Interest-Rate Sensitivity
Script 03: Cross-sectional regressions

A: beta_Inc = alpha + gamma * beta_Exp + e
B: beta_ROA = alpha + gamma * beta_Exp + e

- OLS with heteroskedasticity-robust (HC1) standard errors.
- Size heterogeneity: split banks into terciles by average total assets;
  also an interacted version with demeaned log mean assets.
- Figures: binned scatter plots (20 bins of beta_Exp) with OLS line.

Outputs:
  - output/tables/table2_cross_section.csv
  - output/figures/fig5_inc_vs_exp.png
  - output/figures/fig6_roa_vs_exp.png

Size terciles are defined by each bank's time-series average total assets
(mean_asset), not within-quarter assets.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"
for p in [PROC, OUT_TAB, OUT_FIG]:
    p.mkdir(parents=True, exist_ok=True)

betas = pd.read_csv(f"{PROC}/bank_betas.csv")

# analysis sample: banks with both income and expense betas
A = betas.dropna(subset=["beta_Inc", "beta_Exp"]).copy()
B = betas.dropna(subset=["beta_ROA", "beta_Exp"]).copy()

# log mean assets and size terciles (whole distribution of mean assets)
# NOTE: mean_asset is each bank's TIME-SERIES AVERAGE of total assets over its
# usable sample (computed in script 02), not assets at one point in time. With
# one row per bank, this is the natural size measure; it smooths out quarterly
# swings but also means size is not pinned to a single reference quarter.
for d in (A, B):
    d["log_asset"] = np.log(d["mean_asset"])
    d["log_asset_c"] = d["log_asset"] - d["log_asset"].mean()   # demeaned
    d["size_tercile"] = pd.qcut(d["mean_asset"], 3, labels=["Small", "Medium", "Large"])

# ------------------------------------------------------------------ A
def ols_table(y, x, d, label):
    X = sm.add_constant(d[x])
    m = sm.OLS(d[y], X).fit(cov_type="HC1")
    n = int(m.nobs)
    ar2 = float(m.rsquared_adj)
    return {
        "Specification": label,
        "N": n,
        "Intercept": m.params["const"],
        "Intercept_t": m.tvalues["const"],
        "Slope": m.params[x],
        "Slope_t": m.tvalues[x],
        "Adj_R2": ar2,
    }

rows = [
    ols_table("beta_Inc", "beta_Exp", A, "A: Inc on Exp (full)"),
    ols_table("beta_ROA", "beta_Exp", B, "B: ROA on Exp (full)"),
]
# size terciles - Section A
for t in ["Small", "Medium", "Large"]:
    d = A[A["size_tercile"] == t]
    rows.append(ols_table("beta_Inc", "beta_Exp", d, f"A: Inc on Exp, {t}"))
for t in ["Small", "Medium", "Large"]:
    d = B[B["size_tercile"] == t]
    rows.append(ols_table("beta_ROA", "beta_Exp", d, f"B: ROA on Exp, {t}"))

# interaction with size (demeaned log assets)
# The expense-beta main effect is evaluated at the sample-average bank size
# because log_asset is demeaned before interacting.
for d, y, tag in [(A, "beta_Inc", "A"), (B, "beta_ROA", "B")]:
    X = sm.add_constant(
        pd.DataFrame(
            {
                "Exp": d["beta_Exp"],
                "Exp_x_logA": d["beta_Exp"] * d["log_asset_c"],
                "log_asset_c": d["log_asset_c"],
            }
        )
    )
    m = sm.OLS(d[y], X).fit(cov_type="HC1")
    rows.append(
        {
            "Specification": f"{tag}: interaction with log assets",
            "N": int(m.nobs),
            "Intercept": m.params["const"],
            "Intercept_t": m.tvalues["const"],
            "Slope": m.params["Exp"],
            "Slope_t": m.tvalues["Exp"],
            "Int_coef": m.params["Exp_x_logA"],
            "Int_t": m.tvalues["Exp_x_logA"],
            "Adj_R2": float(m.rsquared_adj),
        }
    )
# ensure the plain rows have the extra columns too
for r in rows:
    r.setdefault("Int_coef", np.nan)
    r.setdefault("Int_t", np.nan)

t2 = pd.DataFrame(rows).round(3)
t2.to_csv(f"{OUT_TAB}/table2_cross_section.csv", index=False)
print(t2.to_string(index=False))

# ------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

def binned_scatter(d, y, x, ax, title, nbins=20):
    d = d.dropna(subset=[y, x])
    # 20 equal-frequency bins of x
    d = d.assign(bin=pd.qcut(d[x], nbins, duplicates="drop"))
    g = d.groupby("bin", observed=True).agg(xm=(x, "mean"), ym=(y, "mean"), n=(y, "size"))
    ax.scatter(g["xm"], g["ym"], s=28, color="steelblue", label="Binned means")
    X = sm.add_constant(d[x])
    m = sm.OLS(d[y], X).fit(cov_type="HC1")
    xs = np.linspace(d[x].min(), d[x].max(), 100)
    ax.plot(xs, m.params["const"] + m.params[x] * xs, color="crimson", lw=1.5,
            label=f"OLS slope {m.params[x]:.2f} (HC1 t={m.tvalues[x]:.1f})")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
binned_scatter(A, "beta_Inc", "beta_Exp", ax,
               "Income beta vs. expense beta (binned)")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/fig5_inc_vs_exp.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
binned_scatter(B, "beta_ROA", "beta_Exp", ax,
               "ROA beta vs. expense beta (binned)")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/fig6_roa_vs_exp.png", bbox_inches="tight")
plt.close(fig)
print("figures saved.")
