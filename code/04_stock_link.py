"""
Bank Interest-Rate Sensitivity
Script 04: Linking banks to CRSP and stock-price analysis

Steps:
 1. Map call-report banks to CRSP PERMCO via Bank_and_CRSP_Linking_Table.csv,
    matching on idrssd first and bhc_id second, restricted to the link
    validity window [dt_start, dt_end]. Report the number of matched banks
    and distinct listed companies, and the number of bank-quarters that map
    to more than one PERMCO (I checked these; see README).
 2. Aggregate multiple PERMNO share classes of the same company-month into
    one company-month record using market-capitalization weights.
 3. Build an equal-weighted and a value-weighted TOTAL RETURN index for
    publicly traded banks (2000-2017, the overlap of CRSP and call reports).
    The index uses MthRet, which includes dividends.
 4. Aggregate bank accounting data to the listed-company (PERMCO) level
    (sum assets, income and expense flows; recompute and quarter-winsorize rates) and regress
    monthly stock returns on LAGGED company accounting levels (not on
    company-level betas, which would embed post-return information).

Accounting-to-return timing: a quarter-q report is treated as public from
two months after the quarter end (Call Report filing lags). Returns earned
before that date are matched to the previous quarter's data, ensuring
accounting data used in a regression were already public.

The company-level expense beta is not used in return regressions because a
full-sample beta would embed post-return information; return tests use only
lagged accounting levels.

Outputs:
  - data_processed/crsp_link_summary.csv
  - data_processed/company_panel.parquet
  - output/tables/table4_stock_regressions.csv
  - output/figures/fig7_bank_stock_index.png
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = ROOT / "data_processed"
OUT_TAB = ROOT / "output" / "tables"
OUT_FIG = ROOT / "output" / "figures"
for p in [PROC, OUT_TAB, OUT_FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
bank = pd.read_parquet(f"{PROC}/bank_panel.parquet")
lk = pd.read_csv(f"{DATA}/Bank_and_CRSP_Linking_Table.csv")

# Read the raw CRSP CSV directly (not a cached parquet), keeping only the
# columns needed for the analysis. ~2.4M rows; the read takes a minute or two.
CRSP_COLS = ["PERMNO", "PERMCO", "YYYYMM", "MthCalDt", "MthPrc", "MthRet",
             "MthCap", "MthVol", "vwretd", "IssuerNm"]
cr = pd.read_csv(f"{DATA}/CRSP_Monthly_Stock_File.csv", usecols=CRSP_COLS)

# PERMCO as a robust integer (some rows can be blank/non-numeric in CRSP)
cr["permco"] = pd.to_numeric(cr["PERMCO"], errors="coerce")
cr = cr.dropna(subset=["permco"]).copy()
cr["permco"] = cr["permco"].astype("int64")
cr["month_dt"] = pd.to_datetime(cr["YYYYMM"].astype(str) + "01")

# ---------------------------------------------------------------- link
# Priority rule: match on the bank's own idrssd first; only banks that do NOT
# match on idrssd are then tried via their holding company's bhc_id. This
# avoids a bank being mapped to two different listed companies (once through
# its own idrssd, once through its BHC), which would double-count its assets
# and flows when aggregating to the company level.
lk["dt_start"] = pd.to_datetime(lk["dt_start"])
lk["dt_end"] = pd.to_datetime(lk["dt_end"])

bank_key = bank[["idrssd", "bhc_id", "qdate", "qdate_dt"]].drop_duplicates()

m1 = bank_key.merge(
    lk.rename(columns={"rssd9001": "idrssd"}), on="idrssd", how="inner")
m1 = m1[(m1["qdate_dt"] >= m1["dt_start"]) & (m1["qdate_dt"] <= m1["dt_end"])]
m1["match_type"] = "idrssd"

# Try the holding-company identifier for bank-quarters without a valid
# own-id link. The fallback is quarter-specific: an own-id match in one period
# must not block a valid BHC match in another period.
own_keys = m1[["idrssd", "qdate"]].drop_duplicates()
bank_bhc = bank_key.merge(own_keys, on=["idrssd", "qdate"], how="left", indicator=True)
bank_bhc = bank_bhc.loc[bank_bhc["_merge"].eq("left_only")].drop(columns="_merge")
bank_bhc = bank_bhc.dropna(subset=["bhc_id"]).rename(columns={"bhc_id": "rssd9001"})
m2 = bank_bhc.merge(lk, on="rssd9001", how="inner").rename(columns={"rssd9001": "bhc_id"})
m2 = m2[(m2["qdate_dt"] >= m2["dt_start"]) & (m2["qdate_dt"] <= m2["dt_end"])]
m2["match_type"] = "bhc_id"

merged = pd.concat([m1[["idrssd", "qdate", "permco", "match_type"]],
                    m2[["idrssd", "qdate", "permco", "match_type"]]], ignore_index=True)

# Prefer own-id links over BHC links. If multiple PERMCOs remain at the same
# priority for one bank-quarter, the link is ambiguous, so exclude that link
# instead of assigning the bank to an arbitrary company.
conf = merged.groupby(["idrssd", "qdate"])["permco"].nunique()
n_conflict_q = int((conf > 1).sum())
print("bank-quarters with >1 PERMCO before resolution:", n_conflict_q)
merged["priority"] = merged["match_type"].map({"idrssd": 0, "bhc_id": 1})
best = merged.groupby(["idrssd", "qdate"])["priority"].transform("min")
best_matches = merged.loc[merged["priority"].eq(best)].copy()
ambiguous = best_matches.groupby(["idrssd", "qdate"])["permco"].nunique()
ambiguous_keys = ambiguous[ambiguous > 1].reset_index()[["idrssd", "qdate"]]
best_matches = best_matches.merge(ambiguous_keys.assign(_ambiguous=True),
                                   on=["idrssd", "qdate"], how="left")
best_matches = best_matches.loc[best_matches["_ambiguous"].isna()]
merged = best_matches.drop_duplicates(["idrssd", "qdate"])[
    ["idrssd", "qdate", "permco", "match_type"]
].copy()
n_ambiguous_q = len(ambiguous_keys)
print("ambiguous bank-quarters excluded:", n_ambiguous_q)

n_banks_matched = merged["idrssd"].nunique()
n_permco = merged["permco"].nunique()
print(f"matched bank-quarters: {len(merged)}, banks: {n_banks_matched}, permco: {n_permco}")
pd.DataFrame({"metric": ["banks_matched", "permco_distinct", "bank_quarter_links",
                         "conflict_bank_quarters", "ambiguous_bank_quarters_excluded"],
              "value": [n_banks_matched, n_permco, len(merged), n_conflict_q, n_ambiguous_q]}
             ).to_csv(f"{PROC}/crsp_link_summary.csv", index=False)

# ------------------------------------------------------- stock returns
# Collapse multiple PERMNO share classes into a company return using the
# beginning-of-month market capitalization weights. Contemporaneous weights
# would use information realized during the return month.
dup = cr.duplicated(["permco", "month_dt"]).sum()
print("permco-month duplicates:", dup)
cr["MthCap"] = pd.to_numeric(cr["MthCap"], errors="coerce")
cr["month_period"] = cr["month_dt"].dt.to_period("M")
company_cap = cr.groupby(["permco", "month_dt"], as_index=False)["MthCap"].sum()
cr = cr.sort_values(["PERMNO", "month_dt"])
cr["cap_lag"] = cr.groupby("PERMNO")["MthCap"].shift(1)
cr["prior_month"] = cr.groupby("PERMNO")["month_period"].shift(1)
cr = cr.loc[cr["prior_month"] + 1 == cr["month_period"]].copy()
cr_d = cr.dropna(subset=["MthCap", "MthRet", "cap_lag"])
cr_d = cr_d.loc[cr_d["cap_lag"] > 0].copy()
cr_d["weighted_ret"] = cr_d["MthRet"] * cr_d["cap_lag"]
comp_month = cr_d.groupby(["permco", "month_dt"]).agg(
    weighted_ret=("weighted_ret", "sum"),
    cap_lag=("cap_lag", "sum"),
    vwretd=("vwretd", "first"),
    IssuerNm=("IssuerNm", "first"),
).reset_index()
comp_month["MthRet"] = comp_month["weighted_ret"] / comp_month["cap_lag"]
comp_month = comp_month.merge(company_cap, on=["permco", "month_dt"], how="left")
comp_month["month_period"] = comp_month["month_dt"].dt.to_period("M")
comp_month = comp_month.sort_values(["permco", "month_dt"])
comp_month["prior_month"] = comp_month.groupby("permco")["month_period"].shift(1)
comp_month["cap_lag_company"] = comp_month.groupby("permco")["MthCap"].shift(1)
comp_month.loc[comp_month["prior_month"] + 1 != comp_month["month_period"],
               "cap_lag_company"] = np.nan

stk = comp_month.copy()
stk = stk.dropna(subset=["MthRet"])
stk = stk[(stk["month_dt"] >= "2000-01-01") & (stk["month_dt"] <= "2017-12-31")]

# -------------------------------------------------------- price index
# MthRet is the total (dividend-inclusive) monthly return, so the compounded
# index is a total-return index.
bank_permcos = merged["permco"].unique()
bstk = stk[stk["permco"].isin(bank_permcos)].copy().sort_values(["permco", "month_dt"])
ew = bstk.groupby("month_dt")["MthRet"].mean().sort_index()
bstk2 = bstk.sort_values(["permco", "month_dt"])
bstk2["cap_lag"] = bstk2["cap_lag_company"]
def wsum(d):
    w = d["cap_lag"] / d["cap_lag"].sum()
    return (w * d["MthRet"]).sum()
vw = (bstk2.dropna(subset=["cap_lag"]).groupby("month_dt").apply(wsum, include_groups=False).sort_index())

idx_ew = (1 + ew).cumprod() * 100
idx_vw = (1 + vw).cumprod() * 100

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(idx_ew.index, idx_ew.values, label="Equal-weighted", lw=1.3)
ax.plot(idx_vw.index, idx_vw.values, label="Value-weighted", lw=1.3)
ax.set_title("Total return index of publicly traded banks (2000 = 100, dividends included)")
ax.set_ylabel("Index")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/fig7_bank_stock_index.png", bbox_inches="tight")
plt.close(fig)
print(f"index: n months {len(idx_ew)}, last EW {idx_ew.iloc[-1]:.0f}, VW {idx_vw.iloc[-1]:.0f}")

# ------------------------------------------------- company aggregation
acc = bank[["idrssd", "qdate", "qdate_dt", "ave_asset", "total_asset",
            "interest_income_q", "interest_expense_q", "net_income_q"]]
acc_m = acc.merge(merged[["idrssd", "qdate", "permco"]], on=["idrssd", "qdate"], how="inner")
comp = (acc_m.groupby(["permco", "qdate", "qdate_dt"]).agg(
    ave_asset=("ave_asset", "sum"), total_asset=("total_asset", "sum"),
    interest_income_q=("interest_income_q", "sum"),
    interest_expense_q=("interest_expense_q", "sum"),
    net_income_q=("net_income_q", "sum"), n_banks=("idrssd", "nunique"),
).reset_index())
comp["IncRate"] = 400.0 * comp["interest_income_q"] / comp["ave_asset"]
comp["ExpRate"] = 400.0 * comp["interest_expense_q"] / comp["ave_asset"]
comp["ROA"] = 400.0 * comp["net_income_q"] / comp["ave_asset"]
# Match the bank-level treatment after recomputing consolidated company rates.
# The cutoffs are cross-sectional within quarter and affect Table 4 regressors.
for v in ["IncRate", "ExpRate", "ROA"]:
    comp[v] = comp.groupby("qdate")[v].transform(
        lambda s: s.clip(s.quantile(0.005), s.quantile(0.995))
    )
comp["log_asset"] = np.log(comp["total_asset"])
comp.to_parquet(f"{PROC}/company_panel.parquet")

# ------------------------------------------------------ public timing
# quarter-end -> first month when report is assumed public (quarter end + 2 months)
comp["public_from"] = comp["qdate_dt"] + pd.offsets.MonthEnd(2) + pd.offsets.MonthBegin(1)
comp["public_ym"] = comp["public_from"].dt.year * 100 + comp["public_from"].dt.month
stk2 = stk.rename(columns={"month_dt": "month"}).copy()
stk2["permco"] = stk2["permco"].astype("int64")
stk2["month_ym"] = stk2["month"].dt.year * 100 + stk2["month"].dt.month

# keep only bank stocks for the asof match
bstk_panel = stk2[stk2["permco"].isin(merged["permco"].unique())].copy()
comp2 = comp[["permco", "public_ym", "log_asset", "IncRate", "ExpRate", "ROA",
              "n_banks"]].copy()

# manual asof: for each permco, latest public_from <= month
def asof_match(stk_g, comp_g):
    c = comp_g.sort_values("public_ym")
    cp = c["public_ym"].to_numpy()
    idx = np.searchsorted(cp, stk_g["month_ym"].to_numpy(), side="right") - 1
    ok = idx >= 0
    out = pd.DataFrame(index=stk_g.index)
    for col in ["log_asset", "IncRate", "ExpRate", "ROA", "n_banks"]:
        out[col] = np.nan
        out.loc[ok, col] = c[col].to_numpy()[idx[ok]]
    return out

parts = []
for permco, g in bstk_panel.groupby("permco", sort=False):
    cg = comp2[comp2["permco"] == permco]
    if len(cg) == 0:
        continue
    m = asof_match(g, cg)
    g2 = g.copy()
    for col in ["log_asset", "IncRate", "ExpRate", "ROA", "n_banks"]:
        g2[col] = m[col].to_numpy()
    parts.append(g2)

panel = pd.concat(parts, ignore_index=True)
panel = panel.dropna(subset=["log_asset", "MthRet"])
print("panel rows:", len(panel))

# -------------------------------------------------------- regressions
# Wide table: one row per variable, one column per specification; each cell
# shows coef (t). Only lagged accounting LEVELS enter (not company betas).
VARS = ["const", "vwretd", "log_asset", "ROA", "IncRate", "ExpRate"]
VLAB = {"const": "Constant", "vwretd": "Market return (vwretd)",
        "log_asset": "Log total assets", "ROA": "ROA (%)",
        "IncRate": "Income rate (%)", "ExpRate": "Expense rate (%)"}

def run(y, X, d, label):
    m = sm.OLS(d[y], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": d["permco"]})
    return {"Spec": label, "N": int(m.nobs), "params": m.params, "tvalues": m.tvalues,
            "R2": float(m.rsquared)}

specs = []
d = panel.dropna(subset=["vwretd"])
specs.append(run("MthRet", d[["vwretd"]], d, "(1) Market"))
X = d[["vwretd", "log_asset", "ROA"]].dropna()
dd = d.loc[X.index]
specs.append(run("MthRet", dd[["vwretd", "log_asset", "ROA"]], dd, "(2) +Size+ROA"))
X3 = dd[["vwretd", "log_asset", "ROA", "IncRate", "ExpRate"]].dropna()
d3 = dd.loc[X3.index]
specs.append(run("MthRet", d3[["vwretd", "log_asset", "ROA", "IncRate", "ExpRate"]], d3, "(3) +Rates"))

for s in specs:
    print(s)

wide = pd.DataFrame({"Variable": [VLAB[v] for v in VARS]})
for s in specs:
    col = []
    for v in VARS:
        if v not in s["params"]:
            col.append("")
        else:
            col.append(f"{s['params'][v]:.4f} ({s['tvalues'][v]:.2f})")
    wide[s["Spec"]] = col
wide.loc[len(wide)] = ["N"] + [str(s["N"]) for s in specs]
wide.loc[len(wide)] = ["R-squared"] + [f"{s['R2']:.4f}" for s in specs]
wide.to_csv(f"{OUT_TAB}/table4_stock_regressions.csv", index=False)
print(wide.to_string(index=False))
print("done.")
