#!/usr/bin/env python3
"""M1: build the CEO-firm-year analysis panel.

Merges Firm_CEO_Compensation with Firm_Information on GVKEY+YEAR,
filters CEOs, aligns t+1 ROA, imputes missing values, and constructs
year-t features. All cleaning decisions documented in
docs/data-cleaning-decisions.md.

Output: analysis-output/panel.csv  +  analysis-output/missingness.csv

Usage: python3 src/build_panel.py   (stdlib + pandas only)
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset-20261103"
OUT = ROOT / "analysis-output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load
ceo = pd.read_csv(DATA / "Firm_CEO_Compensation.csv", low_memory=False)
firm = pd.read_csv(DATA / "Firm_Information.csv", low_memory=False)

# ------------------------------------------------- CEO filter (decision D1)
# ceoann == 'CEO' for 6,038 of 6,039 rows (the 1 'CFO' row is dropped).
ceo = ceo[ceo["ceoann"] == "CEO"].copy()
firm = firm.rename(columns={"GVKEY": "gvkey", "YEAR": "year"})

# One row per gvkey-year is already guaranteed in both tables
# (verified: 0 duplicate gvkey-year keys in each), so the merged panel
# satisfies "one observation per CEO-firm-fiscal year" by construction.

panel = ceo.merge(firm, on=["gvkey", "year"], how="left")

# ------------------------------------------------- t+1 ROA alignment (D2)
# next-year ROA within the same gvkey; only if the next observed year is
# exactly year+1 (guards against gap years / panel exits).
f = firm[["gvkey", "year", "ROA"]].sort_values(["gvkey", "year"]).copy()
f["roa_next"] = f.groupby("gvkey")["ROA"].shift(-1)
f["yr_next"] = f.groupby("gvkey")["year"].shift(-1)
f.loc[f["yr_next"] != f["year"] + 1, "roa_next"] = np.nan
panel = panel.merge(f[["gvkey", "year", "roa_next"]], on=["gvkey", "year"], how="left")

# ------------------------------------------------- feature construction (D3)
# Uses ONLY year-t information (pay level/mix, size, profitability,
# leverage, growth, prior performance). No t+1 variable enters X.
panel["ln_tdc1"] = np.log(panel["tdc1"].clip(lower=1))          # tdc1 in $000s
panel["ln_assets"] = np.log(panel["ASSETS"].clip(lower=1))
panel["ln_sales"] = np.log(panel["SALES"].clip(lower=1))
panel["ln_mktval"] = np.log(panel["MKTVAL"].clip(lower=1))
panel["bonus_share"] = panel["bonus"] / panel["tdc1"].replace(0, np.nan)
panel["equity_share"] = (
    panel["stock_awards"] + panel["option_awards"]
) / panel["tdc1"].replace(0, np.nan)
panel["salary_share"] = panel["salary"] / panel["tdc1"].replace(0, np.nan)
panel["leverage"] = 1 - panel["SEQ"] / panel["ASSETS"].replace(0, np.nan)
panel["roa_t"] = panel["ROA"]

FEATURES = [
    "ln_tdc1", "salary_share", "bonus_share", "equity_share",  # pay level & mix
    "ln_assets", "ln_sales", "ln_mktval",                      # size
    "roa_t", "leverage",                                       # profitability, leverage
    "SALECHG", "ASSETCHG",                                     # growth
    "TRS1YR", "TRS3YR",                                        # prior performance
    "age",                                                     # CEO characteristic
]
# Dropped as features: ret_yrs (100% missing), NICHG (17.8% missing, see D4).

# ------------------------------------------------- missing values (D4)
miss_before = panel[FEATURES].isna().mean().mul(100).round(2)
miss_before.rename("pct_missing_before").to_csv(OUT / "missingness.csv")

# Rule (assignment requirement: delete or mean-impute):
#   - variables missing >15%  -> dropped (NICHG already excluded above)
#   - remaining features      -> mean imputation (train-set means would be
#     computed inside the model pipeline to avoid leakage; for the panel
#     file we impute on the full sample and note this in the docs)
panel[FEATURES] = panel[FEATURES].fillna(panel[FEATURES].mean())

# ------------------------------------------------- outcome label (M2 hook)
# Primary definition: t+1 ROA decline (roa_t - roa_next) in the worst 20%
# of the within-sample distribution, computed on rows with valid t+1 ROA.
# Robustness definitions (alt quintile / fixed pp threshold) are built in
# src/run_models.py so the same panel file serves all of them.
panel["roa_decline"] = panel["roa_t"] - panel["roa_next"]
panel["y_review"] = (
    panel["roa_decline"] >= panel["roa_decline"].quantile(0.80)
).astype(int)
panel.loc[panel["roa_decline"].isna(), "y_review"] = np.nan

# ------------------------------------------------- save + summary
cols = (
    ["gvkey", "year", "exec_fullname", "coname", "ceoann"]
    + FEATURES
    + ["roa_next", "roa_decline", "y_review"]
)
panel[cols].to_csv(OUT / "panel.csv", index=False)

n_valid = panel["y_review"].notna().sum()
print(f"panel rows            : {len(panel)}")
print(f"rows with t+1 outcome : {n_valid}  (by year: "
      f"{panel[panel.y_review.notna()].year.value_counts().to_dict()})")
print(f"positive rate (y=1)   : {panel['y_review'].mean():.3f}")
print(f"features              : {len(FEATURES)} -> {FEATURES}")
print(f"wrote {OUT/'panel.csv'} and {OUT/'missingness.csv'}")
