"""
Bank Interest-Rate Sensitivity
Script 02b: Asymmetric expense-rate betas

This script tests whether the pass-through of policy-rate changes to bank
funding costs is symmetric in rate increases versus rate decreases. I split
the quarterly federal-funds change into an increase part and a decrease part,

    dff_pos = max(dff, 0),   dff_neg = min(dff, 0),

and for each bank estimate

    d_ExpRate_it = a_i + sum_{tau=0..3} b_up_itau * dff_pos_{t-tau}
                         + sum_{tau=0..3} b_down_itau * dff_neg_{t-tau} + u_it

    beta_Exp_up   = sum_tau b_up_itau
    beta_Exp_down = sum_tau b_down_itau

Both sums capture the cumulative response of the funding-cost rate to four
quarterly changes in the policy rate, exactly as in Script 02, but separately
for rate hikes and cuts.

Only adjacent-quarter differences of ExpRate are used (same rule as script
02). A bank needs at least N_MIN usable observations to keep its estimates;
I use 28 (7 years) rather than 20 because splitting the regressors roughly
halves the effective variation on each side. Estimated betas are not trimmed.

The script runs the same estimation twice:
  - the full sample (1984Q1-2017Q4), saved as bank_betas_asym.csv;
  - a pre-2007 sample that truncates the panel at 2007Q4 (qdate SAMPLE_END)
    before constructing any difference, saved as bank_betas_asym_pre07.csv.
The pre-2007 version exists because the asymmetric beta estimated on the
full sample would embed post-2007 information (including the zero-lower-bound
period) and could not separately identify whether the asymmetry predates
2008. Truncating the data before estimating avoids that look-ahead: the
pre-2007 betas use only rate changes observed through 2007Q4.

Outputs:
  - data_processed/bank_betas_asym.csv        (full sample)
  - data_processed/bank_betas_asym_pre07.csv  (qdate <= 2007Q4 only)
  both with columns (idrssd, beta_Exp_up, beta_Exp_down, n_obs)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
PROC.mkdir(parents=True, exist_ok=True)

N_MIN = 28       # minimum usable observations for the asymmetric regression
SAMPLE_END = 2007 * 4 + 3  # calendar-quarter index for 2007Q4

df = pd.read_parquet(f"{PROC}/bank_panel.parquet")
df = df.sort_values(["idrssd", "qdate"]).reset_index(drop=True)


def estimate_sample(sample, out_path, tag):
    """Run the asymmetric regression on one truncated panel and save it."""
    sample = sample.copy()

    # --- quarterly fed funds changes within this sample, split pos / neg --
    ff_by_q = sample.drop_duplicates("qindex")[["qindex", "ffrate"]].set_index("qindex")["ffrate"]
    calendar_q = np.arange(int(sample["qindex"].min()), int(sample["qindex"].max()) + 1)
    ff_by_q = ff_by_q.reindex(calendar_q)
    dff = ff_by_q.diff()
    dff_pos = dff.clip(lower=0.0)
    dff_neg = dff.clip(upper=0.0)
    for lag in range(4):
        sample[f"dff_pos_l{lag}"] = sample["qindex"].map(dff_pos.shift(lag))
        sample[f"dff_neg_l{lag}"] = sample["qindex"].map(dff_neg.shift(lag))

    # --- first differences of ExpRate within bank (adjacent quarters only)
    dq = sample.groupby("idrssd")["qindex"].diff()
    adjacent = dq == 1
    prev = sample.groupby("idrssd")["ExpRate_w"].shift(1)
    sample["d_ExpRate_w"] = np.where(
        adjacent & prev.notna(), sample["ExpRate_w"] - prev, np.nan
    )

    rows = []
    for bid, b in sample.groupby("idrssd", sort=False):
        y = b["d_ExpRate_w"].to_numpy(float)
        X = np.column_stack([
            np.ones(len(b)),
            b["dff_pos_l0"].to_numpy(float), b["dff_pos_l1"].to_numpy(float),
            b["dff_pos_l2"].to_numpy(float), b["dff_pos_l3"].to_numpy(float),
            b["dff_neg_l0"].to_numpy(float), b["dff_neg_l1"].to_numpy(float),
            b["dff_neg_l2"].to_numpy(float), b["dff_neg_l3"].to_numpy(float),
        ])
        ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        y, X = y[ok], X[ok]
        n = len(y)
        out = {"idrssd": bid, "beta_Exp_up": np.nan, "beta_Exp_down": np.nan, "n_obs": n}
        if n < N_MIN or np.linalg.matrix_rank(X) < 9:
            rows.append(out)
            continue
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["beta_Exp_up"] = coef[1:5].sum()
        out["beta_Exp_down"] = coef[5:9].sum()
        rows.append(out)

    asym = pd.DataFrame(rows)
    asym.to_csv(out_path, index=False)
    ok = asym.dropna(subset=["beta_Exp_up", "beta_Exp_down"])
    print(f"[{tag}] banks: {len(asym)}, with both betas: {len(ok)}, "
          f"median up {ok['beta_Exp_up'].median():.3f}, "
          f"median down {ok['beta_Exp_down'].median():.3f}")
    print("saved", out_path)


if __name__ == "__main__":
    estimate_sample(df, f"{PROC}/bank_betas_asym.csv", "full sample")
    pre07 = df[df["qindex"] <= SAMPLE_END].copy()
    estimate_sample(pre07, f"{PROC}/bank_betas_asym_pre07.csv", "pre-2007")
