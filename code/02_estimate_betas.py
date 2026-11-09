"""
Bank Interest-Rate Sensitivity
Script 02: Bank-level interest income / expense / ROA betas

For bank i and quarter t:
    d_y = y_it - y_i,t-1            (first difference within bank)
    d_f = f_t - f_t-1               (federal funds rate, quarter-end month)

    d_y_it = a_i + sum_{tau=0..3} b_i,tau * d_f_{t-tau} + u_it
    beta_i = b_i0 + b_i1 + b_i2 + b_i3

y in {IncRate_w, ExpRate_w, ROA_w} (winsorized annualized % rates).
Eligibility: bank must have >= N_MIN usable d_y observations
(calendar-quarter contiguous diffs with non-missing y_t and y_{t-1}).

Outputs:
  - data_processed/bank_betas.csv        (one row per eligible bank)
  - output/tables/beta_summary.csv       (summary of bank-level estimates)
  - output/figures/fig4_beta_distributions.png

Note on naming: the internal columns keep the _w suffix (they were estimated
from winsorized rates), but the output CSV and the report use beta_Inc,
beta_Exp, beta_ROA. The cleaned dataset has "one row for each bank and that bank's
estimated interest income and interest expense betas"; I keep all banks in the
file, with betas missing for banks that do not meet the observation minimum.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"
for p in [PROC, OUT_TAB, OUT_FIG]:
    p.mkdir(parents=True, exist_ok=True)

N_MIN = 20   # minimum usable quarterly observations per bank (5 years)
YVARS = ["IncRate_w", "ExpRate_w", "ROA_w"]
# internal -> public names for the beta columns
BETA_NAMES = {"IncRate_w": "beta_Inc", "ExpRate_w": "beta_Exp", "ROA_w": "beta_ROA"}

# ---------------------------------------------------------------- load
df = pd.read_parquet(f"{PROC}/bank_panel.parquet")
df = df.sort_values(["idrssd", "qdate"]).reset_index(drop=True)

# --- quarterly fed funds changes (common across banks) ----------------
ff_by_q = df.drop_duplicates("qindex")[["qindex", "ffrate"]].set_index("qindex")["ffrate"]
calendar_q = np.arange(int(df["qindex"].min()), int(df["qindex"].max()) + 1)
ff_by_q = ff_by_q.reindex(calendar_q)
dff = ff_by_q.diff()
# attach lags 0..3 to every bank-quarter
for lag in range(4):
    df[f"dff_l{lag}"] = df["qindex"].map(dff.shift(lag))

# --- first differences of y within bank ---------------------------------
# IMPORTANT: only keep a first difference if the two quarters are ADJACENT
# (qindex difference == 1). Banks sometimes skip a quarter; a naive shift(1)
# difference would then span two quarters of y while the fed funds change
# dff_l0 still covers one quarter, mismatching the two sides of the
# regression. I found this by spotting a few banks whose d_y jumped by more
# than any single-quarter rate change could explain.
dq = df.groupby("idrssd")["qindex"].diff()
adjacent = dq == 1
for v in YVARS:
    prev = df.groupby("idrssd")[v].shift(1)
    df[f"d_{v}"] = np.where(adjacent & prev.notna(), df[v] - prev, np.nan)

# --- per-bank regression -------------------------------------------------
bank_ids = df["idrssd"].unique()

def fit_bank(bid):
    b = df[df["idrssd"] == bid]
    res = {"idrssd": bid}
    for v in YVARS:
        y = b[f"d_{v}"].to_numpy(float)
        X = np.column_stack([
            np.ones(len(b)),
            b["dff_l0"].to_numpy(float),
            b["dff_l1"].to_numpy(float),
            b["dff_l2"].to_numpy(float),
            b["dff_l3"].to_numpy(float),
        ])
        ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        y, X = y[ok], X[ok]
        n = len(y)
        res[f"n_{v}"] = n
        if n < N_MIN or np.linalg.matrix_rank(X) < 5:
            res[f"beta_{v}"] = np.nan
            res[f"r2_{v}"] = np.nan
            continue
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        beta = coef[1:5].sum()
        # R2
        yhat = X @ coef
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        res[f"beta_{v}"] = beta
        res[f"r2_{v}"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return res

if __name__ == "__main__":
    # bank attributes for reporting
    attrs = (
        df.groupby("idrssd")
        .agg(
            n_quarters=("qdate", "count"),
            first_q=("qdate", "min"),
            last_q=("qdate", "max"),
            mean_asset=("total_asset", "mean"),
            median_asset=("total_asset", "median"),
            max_asset=("total_asset", "max"),
            bank_name=("bank_name", "last"),
            bhc_id=("bhc_id", "last"),
        )
        .reset_index()
    )

    with Pool(8) as pool:
        rows = pool.map(fit_bank, bank_ids, chunksize=200)

    betas = pd.DataFrame(rows).merge(attrs, on="idrssd", how="left")

    # rename beta columns to the public names (drop _w in output/report)
    betas = betas.rename(columns={
        f"beta_{v}": BETA_NAMES[v] for v in YVARS
    })
    # keep n_* / r2_* under public names as well for readability
    betas = betas.rename(columns={
        f"n_{v}": f"n_{BETA_NAMES[v]}" for v in YVARS
    })

    # eligibility for each outcome: at least one beta non-missing
    for v in YVARS:
        betas[f"eligible_{BETA_NAMES[v]}"] = betas[BETA_NAMES[v]].notna()
    betas["eligible_any"] = betas[[f"eligible_{BETA_NAMES[v]}" for v in YVARS]].any(axis=1)

    betas.to_csv(f"{PROC}/bank_betas.csv", index=False)
    print("banks with any beta:", betas["eligible_any"].sum(), "/", len(betas))
    for v in YVARS:
        print(f"  eligible_{BETA_NAMES[v]}: {betas[f'eligible_{BETA_NAMES[v]}'].sum()}")

    # ---------------------------------------------------------- summary
    summ = pd.DataFrame(
        {
            "Statistic": ["N banks", "Mean", "SD", "p10", "p25", "Median", "p75", "p90"],
            "Beta_Inc": [
                int(betas["beta_Inc"].notna().sum()),
                betas["beta_Inc"].mean(),
                betas["beta_Inc"].std(),
                betas["beta_Inc"].quantile(0.10),
                betas["beta_Inc"].quantile(0.25),
                betas["beta_Inc"].median(),
                betas["beta_Inc"].quantile(0.75),
                betas["beta_Inc"].quantile(0.90),
            ],
            "Beta_Exp": [
                int(betas["beta_Exp"].notna().sum()),
                betas["beta_Exp"].mean(),
                betas["beta_Exp"].std(),
                betas["beta_Exp"].quantile(0.10),
                betas["beta_Exp"].quantile(0.25),
                betas["beta_Exp"].median(),
                betas["beta_Exp"].quantile(0.75),
                betas["beta_Exp"].quantile(0.90),
            ],
            "Beta_ROA": [
                int(betas["beta_ROA"].notna().sum()),
                betas["beta_ROA"].mean(),
                betas["beta_ROA"].std(),
                betas["beta_ROA"].quantile(0.10),
                betas["beta_ROA"].quantile(0.25),
                betas["beta_ROA"].median(),
                betas["beta_ROA"].quantile(0.75),
                betas["beta_ROA"].quantile(0.90),
            ],
        }
    )
    summ = summ.round(3)
    # Keep the count row as an integer when serialized for the report table.
    for col in ["Beta_Inc", "Beta_Exp", "Beta_ROA"]:
        summ[col] = summ[col].astype(object)
        summ.at[0, col] = str(int(summ.at[0, col]))
    summ.to_csv(f"{OUT_TAB}/beta_summary.csv", index=False)
    print("\n", summ.to_string(index=False))

    # ------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, v, ttl in zip(
        axes,
        ["beta_Inc", "beta_Exp", "beta_ROA"],
        ["Interest income beta", "Interest expense beta", "ROA beta"],
    ):
        s = betas[v].dropna()
        ax.hist(s, bins=60, color="steelblue", alpha=0.8)
        ax.axvline(s.median(), color="crimson", ls="--", lw=1, label=f"median {s.median():.2f}")
        ax.set_title(ttl)
        ax.set_xlabel("Beta")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig4_beta_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print("done.")
