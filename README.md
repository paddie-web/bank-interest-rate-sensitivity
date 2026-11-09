# Bank Interest-Rate Sensitivity

A reproducible empirical analysis of how U.S. commercial banks' interest income,
interest expense, and profitability respond to changes in the federal funds rate,
using quarterly Call Report data (1984–2017) linked to CRSP for publicly traded
banks. The analysis estimates bank-level interest-rate betas, relates them in the
cross section, tests whether funding-cost pass-through is symmetric in rate hikes
versus cuts, and links accounting rates to stock returns.

The full write-up — every formula in LaTeX, with the complete code reproduced as
an appendix — is a single PDF:

- `output/bank_interest_rate_sensitivity.pdf`

## Repository layout

- `code/` — seven Python scripts covering data preparation, estimation, analysis, and report generation.
- `output/bank_interest_rate_sensitivity.pdf` / `.tex` — the full report, with every table and figure and the complete code reproduced in an appendix.
- `output/figures/` — the figures embedded in the report.
- `data/` and `data_processed/` — raw inputs and generated panels are intentionally excluded (see Data sources). When the licensed source files are present, the scripts regenerate them locally.

## Reproduce

From the project root, run the scripts in order:

```bash
python code/01_data_prep.py
python code/02_estimate_betas.py
python code/02b_asym_betas.py
python code/03_cross_section.py
python code/04_stock_link.py
python code/05_q2_extension.py
python code/06_make_report.py
```

Each script resolves paths relative to the project root and creates its output
folders. Scripts 01–05 read the raw files from `data/` and write
`data_processed/`, `output/tables/`, and `output/figures/`. Script 06 reads the
generated tables and figures, regenerates the LaTeX source, and compiles the PDF
with `pdflatex` (including the code appendix). The generated CSV panels and
tables are not committed here; an end-to-end run therefore requires the licensed
raw files, while the committed PDF and its figures are fully readable on their own.

Requirements: Python 3 with pandas, NumPy, statsmodels, SciPy, matplotlib, and
PyArrow, plus a TeX distribution providing `pdflatex`. Script 01 reads the Stata
input directly through pandas.

## Data sources

The four raw files belong in `data/` and are not redistributed here (CRSP is
licensed):

- `callreports_1984_2017.dta` — quarterly Call Report fields (quarterly average assets, total interest income, total interest expense, net income, and bank/holding-company identifiers), from the FFIEC / Federal Reserve Consolidated Reports of Condition and Income.
- `FedFunds.csv` — monthly effective federal funds rate (Federal Reserve / FRED).
- `Bank_and_CRSP_Linking_Table.csv` — validity-dated mapping of bank and holding-company RSSDs to CRSP PERMCOs.
- `CRSP_Monthly_Stock_File.csv` — CRSP monthly stock returns, market capitalization, and delisting-adjusted returns (CRSP via WRDS).

## Data choices

- The three Call Report income/expense/net-income items are year-to-date flows. Single-quarter flows are recovered by differencing adjacent quarters within each bank-year; Q1 is the reported Q1 value, and a skipped quarter is set to missing rather than filled with a two-quarter change.
- Rates are annualized and winsorized at the 0.5/99.5 percentiles within each quarter before estimation. First differences are retained only across consecutive calendar quarters.
- Bank-level betas require at least 20 usable observations (13,624 of 17,743 banks qualify). The asymmetric pass-through model splits federal-funds changes into positive and negative parts and requires 28 observations; results are reported for the full sample, size terciles, and a pre-2007 re-estimation.
- For publicly traded banks, links are matched by own RSSD first and holding-company RSSD second; multiple CRSP share classes are combined using prior-month market-cap weights. Company rates are recomputed from aggregated flows and assets, winsorized at 0.5/99.5 within quarter, and matched after an assumed two-month reporting lag.

All estimates are descriptive, not causal.
