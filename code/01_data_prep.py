"""
Bank Interest-Rate Sensitivity
Script 01: Data preparation from raw files

Inputs (raw, unmodified):
  - data/callreports_1984_2017.dta
  - data/FedFunds.csv
Outputs:
  - data_processed/bank_panel.parquet
  - output/tables/table1_descriptives.csv
  - output/figures/fig1_industry_over_time.png
  - output/figures/fig2_missing_values.png
  - output/figures/fig3_rate_trends.png

Things I hit while writing this, kept so the reader sees the reasoning:

  * The income/expense/net-income items are year-to-date (YTD) cumulative
    flows. I first built ROA directly from reported "net income" and got a
    seasonal sawtooth: Q4 ROA was systematically higher than Q1 ROA for nearly
    every bank. That is the tell for a YTD variable. I then checked complete
    bank-years: the four reconstructed quarterly flows equal reported Q4 YTD.
  * My first version differenced YTD across any same-year rows, which turned a
    skipped Q2 into a Q3 flow of Q2+Q3. Now I difference only adjacent quarters
    and set skipped quarters to missing.
  * ave_asset is a quarterly-average stock, not a flow, so it is NOT differenced.
  * Missingness is small but non-random over time (more in the 1980s and again
    around 2005-06); I drop those rows rather than impute.
  * I winsorize rates at 0.5/99.5 within quarter before beta estimation so a
    handful of merger quarters do not drive an individual bank's regression.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = ROOT / "data_processed"
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"
for p in [PROC, OUT_TAB, OUT_FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
# Read the raw .dta directly (no cached parquet), keeping only the columns
# used. This guarantees the script runs from the provided raw file.
cols = ["idrssd", "qdate", "year", "quarter", "charter_type", "bhc_id",
        "bank_name", "total_asset", "ave_asset", "interest_income",
        "interest_expense", "net_income"]
df = pd.read_stata(f"{DATA}/callreports_1984_2017.dta", columns=cols, convert_categoricals=False)
df = df.sort_values(["idrssd", "qdate"]).reset_index(drop=True)
df["qindex"] = df["year"].astype(int) * 4 + df["quarter"].astype(int) - 1

# calendar quarter-end date (last day of the quarter); used for the Fed funds
# merge and for the CRSP "public from" timing in script 04.
_qm = {1: 3, 2: 6, 3: 9, 4: 12}
_qd = {1: 31, 2: 30, 3: 30, 4: 31}
df["qdate_dt"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["quarter"].map(_qm).astype(str).str.zfill(2)
    + "-" + df["quarter"].map(_qd).astype(str).str.zfill(2)
)

# --- restore single-quarter flows from YTD items ---------------------
# qindex is a consecutive-quarter integer built from year and quarter. Only
# difference when the previous row for the same bank is the immediately
# preceding calendar quarter (qindex diff == 1). A skipped quarter is set to
# missing, not faked by differencing across the gap.
dq = df.groupby("idrssd")["qindex"].diff()
adjacent = dq == 1
for v in ["interest_income", "interest_expense", "net_income"]:
    prev = df.groupby("idrssd")[v].shift(1)
    is_q1 = df["quarter"] == 1
    df[v + "_q"] = np.where(is_q1, df[v],
                            np.where(adjacent & prev.notna(), df[v] - prev, np.nan))

# Validate the reconstruction only for complete bank-years with all four
# quarters observed and reconstructed. Incomplete years cannot be compared
# with a Q4 YTD total.
annual = df.groupby(["idrssd", "year"]).agg(
    quarters=("quarter", lambda s: tuple(sorted(s.dropna().unique()))),
    n_rows=("quarter", "size"),
)
complete_keys = annual.index[annual["quarters"].map(lambda x: x == (1, 2, 3, 4)) & annual["n_rows"].eq(4)]
for v in ["interest_income", "interest_expense", "net_income"]:
    qsum = df.groupby(["idrssd", "year"])[v + "_q"].sum(min_count=4)
    q4 = df.loc[df["quarter"].eq(4)].set_index(["idrssd", "year"])[v]
    chk = pd.concat([qsum.rename("qsum"), q4.rename("q4_ytd")], axis=1).loc[complete_keys].dropna()
    err = (chk["qsum"] - chk["q4_ytd"]).abs()
    print(f"[check] {v}: complete bank-years={len(chk)}, exact within 1e-6={(err < 1e-6).mean():.3%}, max abs diff={err.max():.6g}")

# ------------------------------------------------------------- rates
# filter first (non-positive or missing asset / flow rows) so the rate
# denominators are strictly positive before dividing.
before = len(df)
raw_banks = df["idrssd"].nunique()
df = df.dropna(subset=["ave_asset", "interest_income_q", "interest_expense_q", "net_income_q"])
df = df[(df["ave_asset"] > 0) & (df["total_asset"] > 0)]
print(f"[drop] rows {before} -> {len(df)}")

df["IncRate"] = 400.0 * df["interest_income_q"] / df["ave_asset"]   # % per year
df["ExpRate"] = 400.0 * df["interest_expense_q"] / df["ave_asset"]
df["ROA"]     = 400.0 * df["net_income_q"] / df["ave_asset"]

# -------------------------------------------------------- Fed funds
ff = pd.read_csv(f"{DATA}/FedFunds.csv")
ff["date"] = pd.to_datetime(ff["date"])
ff = ff.sort_values("date").reset_index(drop=True)
# effective fed funds rate at the end (last month) of each calendar quarter
ff["qend"] = ff["date"].dt.to_period("Q").dt.end_time.dt.normalize()
qff = ff.groupby("qend")["FedFunds"].last().reset_index()
qff.columns = ["qdate_dt", "ffrate"]
df = df.merge(qff, on="qdate_dt", how="left")

# ---------------------------------------------------------------- FE
df["log_asset"] = np.log(df["total_asset"])

# winsorized versions of the rates (0.5/99.5, cross-section by quarter).
# The _w suffix is internal; the report/tables drop it.
def winsor(s, lo=0.005, hi=0.995):
    l, h = s.quantile(lo), s.quantile(hi)
    return s.clip(l, h)

df["IncRate_w"] = df.groupby("qdate")["IncRate"].transform(lambda s: winsor(s))
df["ExpRate_w"] = df.groupby("qdate")["ExpRate"].transform(lambda s: winsor(s))
df["ROA_w"] = df.groupby("qdate")["ROA"].transform(lambda s: winsor(s))

# ------------------------------------------------------------ Table 1
# Assets are reported in $ thousands; I convert to $ millions in the table so
# the magnitudes are readable (median bank ~$66m of assets). Only the main
# variables used downstream are shown (the winsorized rates are the ones used
# in estimation; the raw versions are not reported separately).
tab = pd.DataFrame(
    {
        "Variable": [
            "Ave assets ($m)", "Total assets ($m)", "IncRate (% p.a.)",
            "ExpRate (% p.a.)", "ROA (% p.a.)", "Fed funds rate (%)",
        ],
        "N": [
            df["ave_asset"].count(), df["total_asset"].count(),
            df["IncRate_w"].count(), df["ExpRate_w"].count(), df["ROA_w"].count(),
            df["ffrate"].count(),
        ],
        "Mean": [
            df["ave_asset"].mean() / 1e3, df["total_asset"].mean() / 1e3,
            df["IncRate_w"].mean(), df["ExpRate_w"].mean(), df["ROA_w"].mean(),
            df["ffrate"].mean(),
        ],
        "SD": [
            df["ave_asset"].std() / 1e3, df["total_asset"].std() / 1e3,
            df["IncRate_w"].std(), df["ExpRate_w"].std(), df["ROA_w"].std(),
            df["ffrate"].std(),
        ],
        "p1": [
            df["ave_asset"].quantile(0.01) / 1e3, df["total_asset"].quantile(0.01) / 1e3,
            df["IncRate_w"].quantile(0.01), df["ExpRate_w"].quantile(0.01),
            df["ROA_w"].quantile(0.01), df["ffrate"].quantile(0.01),
        ],
        "Median": [
            df["ave_asset"].median() / 1e3, df["total_asset"].median() / 1e3,
            df["IncRate_w"].median(), df["ExpRate_w"].median(), df["ROA_w"].median(),
            df["ffrate"].median(),
        ],
        "p99": [
            df["ave_asset"].quantile(0.99) / 1e3, df["total_asset"].quantile(0.99) / 1e3,
            df["IncRate_w"].quantile(0.99), df["ExpRate_w"].quantile(0.99),
            df["ROA_w"].quantile(0.99), df["ffrate"].quantile(0.99),
        ],
    }
)
tab = tab.round(3)
tab.to_csv(f"{OUT_TAB}/table1_descriptives.csv", index=False)
print(tab.to_string(index=False))

# ---------------------------------------------------------------- Fig 1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
g = df.groupby("qdate_dt").agg(
    n_banks=("idrssd", "nunique"),
    tot_asset=("total_asset", "sum"),
    med_asset=("total_asset", "median"),
)
axes[0].plot(g.index, g["n_banks"])
axes[0].set_title("Number of reporting banks")
axes[0].set_xlabel("Year")
g2 = df.groupby("qdate_dt")["total_asset"].sum() / 1e9  # $tn (raw unit: $ thousand)
axes[1].plot(g2.index, g2)
axes[1].set_title("Total assets ($tn)")
axes[1].set_xlabel("Year")
axes[2].plot(g.index, g["med_asset"] / 1e3)
axes[2].set_title("Median bank assets ($m)")
axes[2].set_xlabel("Year")
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/fig1_industry_over_time.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 2
# missingness by year (share of bank-quarter rows with missing flows)
# computed on the raw panel BEFORE dropping rows
raw = pd.read_stata(
    f"{DATA}/callreports_1984_2017.dta",
    columns=["year", "interest_income", "interest_expense", "net_income"],
    convert_categoricals=False,
)
raw["any_missing"] = raw[["interest_income", "interest_expense", "net_income"]].isna().any(axis=1)
miss = raw.groupby("year")["any_missing"].mean() * 100
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.bar(miss.index, miss.values)
ax.set_title("Share of bank-quarters with missing income/expense items (%)")
ax.set_xlabel("Year")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/fig2_missing_values.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 3
# One trend figure for the data-audit section: annual median IncRate, ExpRate, NIM, ROA.
med = df.groupby("year")[["IncRate_w", "ExpRate_w", "ROA_w"]].median()
df["NIM_w"] = df["IncRate_w"] - df["ExpRate_w"]
med["NIM"] = df.groupby("year")["NIM_w"].median()
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(med.index, med["IncRate_w"], lw=1.5, label="Median IncRate")
ax.plot(med.index, med["ExpRate_w"], lw=1.5, label="Median ExpRate")
ax.plot(med.index, med["NIM"], lw=1.5, label="Median NIM (IncRate - ExpRate)")
ax.plot(med.index, med["ROA_w"], lw=1.5, ls="--", label="Median ROA")
ax.set_title("Median bank rates over time")
ax.set_ylabel("% per annum")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/fig3_rate_trends.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- save
df.to_parquet(f"{PROC}/bank_panel.parquet", index=False)
pd.DataFrame({
    "metric": ["raw_rows", "raw_banks", "clean_rows", "clean_banks", "dropped_rows"],
    "value": [before, raw_banks, len(df), df["idrssd"].nunique(), before - len(df)],
}).to_csv(PROC / "sample_summary.csv", index=False)
print("\nSaved bank_panel.parquet:", df.shape)
print("Banks:", df["idrssd"].nunique(), " Period:", df["year"].min(), "-", df["year"].max())
