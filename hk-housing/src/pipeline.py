"""
pipeline.py — Build the monthly estimation panel from official raw files.

Sources (required input paths and schemas are listed in input_manifest.json):
  RVD his_data_4.xls  : private domestic price indices by class, monthly 1993+
  RVD his_data_3.xls  : private domestic rental indices by class, monthly 1993+
  RVD private_domestic.xls : completions / stock / vacancy, annual
  C&SD                : composite CPI (monthly), median household income (qtr),
                        mid-year population (annual)
  HKMA API            : BLR & retail rates (monthly, 1980+), 1m HIBOR (1996+),
                        residential mortgage survey (1996+)
  Land Registry       : residential sale & purchase agreements (monthly)
  FRED                : US federal funds rate (monthly)

Output: data/processed/panel.csv — one row per month with raw and constructed
variables; constructions documented inline.

Two constructions matter for interpretation:
  * RVD stock figures exclude village houses from 2003 on. Year-end stock
    before 2003 is rescaled to the new definition with the 2003 ratio
    (end-2003 stock less 2003 completions, over end-2002 stock), so the
    definition change is not read as a fall in supply.
  * Quarterly median income is interpolated to months for the in-sample
    models (linc_real). The forecasts use linc_real_rt instead: each month
    holds the latest quarter treated as published by then (INCOME_LAG_MONTHS
    after the quarter ends), carried forward, so no forecast origin sees a
    later quarter.
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
# months after a quarter ends before its median income is treated as known
# (a conservative assumption about the release timing)
INCOME_LAG_MONTHS = 2

CLASS_COL_CANDIDATES = {
    "A": (7, 8), "B": (10, 11), "C": (13, 14), "D": (16, 17), "E": (19, 20),
    "ABC": (22, 23), "DE": (25, 26), "ALL": (28, 29),
}


def _coerce_idx(v):
    """RVD cell → float. Handles '(58.1)' (<20 transactions), '*', blanks."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace("(", "").replace(")", "").replace("*", "").strip()
    if s in ("", "-", "N.A.", "n.a."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _coerce_year(v):
    """RVD year cells are usually numeric but occasionally strings
    (e.g. the rent file stores 2004 as the text '2004')."""
    if pd.isna(v):
        return None
    try:
        y = float(str(v).strip())
    except ValueError:
        return None
    return int(y) if 1980 <= y <= 2100 else None


def parse_rvd_monthly(path: str, prefix: str) -> pd.DataFrame:
    """Parse an RVD by-class monthly index sheet (price or rent)."""
    df = pd.read_excel(path, sheet_name=0, header=None)
    year = np.nan
    rows = []
    for _, r in df.iterrows():
        y = _coerce_year(r.iloc[1])
        if y is not None:
            year = y
        m = r.iloc[5]
        if pd.isna(m) or pd.isna(year):
            continue
        try:
            m = int(float(m))
        except (TypeError, ValueError):
            continue
        if not 1 <= m <= 12:
            continue
        rec = {"date": pd.Timestamp(year=year, month=m, day=1) + pd.offsets.MonthEnd(0)}
        for cls, cols in CLASS_COL_CANDIDATES.items():
            val = np.nan
            for c in cols:
                if c < len(r):
                    val = _coerce_idx(r.iloc[c])
                    if not np.isnan(val):
                        break
            rec[f"{prefix}_{cls}"] = val
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out = out.dropna(subset=[f"{prefix}_ALL"])
    # hard validation: monthly index must be unique and gap-free
    if out.index.duplicated().any():
        dupes = out.index[out.index.duplicated()].unique()
        raise ValueError(f"{path}: duplicated months after parse: {dupes[:5]}")
    gaps = pd.date_range(out.index.min(), out.index.max(), freq="ME").difference(out.index)
    if len(gaps) > 0:
        raise ValueError(f"{path}: missing months after parse: {gaps[:5]}")
    return out


def parse_rvd_stock(path: str) -> pd.DataFrame:
    """Annual completions and year-end stock (all classes) from RVD tables."""
    res = {}
    for sheet, name in [("Completions_落成量", "completions"), ("Stock_總存量", "stock")]:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        recs = {}
        for _, r in df.iterrows():
            y = r.iloc[2]
            if not (isinstance(y, (int, float)) and not pd.isna(y) and 1980 <= float(y) <= 2100):
                continue
            vals = [_coerce_idx(v) for v in r.iloc[3:]]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                recs[int(y)] = vals[-1]  # rightmost numeric = all-classes total
        res[name] = pd.Series(recs, name=name)
    out = pd.DataFrame(res)
    out.index.name = "year"
    return out


def splice_stock(stock_a: pd.DataFrame) -> tuple[pd.Series, float]:
    """Year-end stock on the 2003+ definition (village houses excluded).

    Years before 2003 are scaled by (stock_2003 − completions_2003) / stock_2002,
    the end-2002 stock the new definition implies. Returns (stock, factor).
    """
    st = stock_a["stock"].dropna()
    factor = float((st.loc[2003] - stock_a.loc[2003, "completions"]) / st.loc[2002])
    return st.where(st.index >= 2003, st * factor), factor


def load_cpi() -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, "censtatd", "censtatd_cpi_composite_monthly.csv"))
    s = pd.Series(df["cpi_composite_index"].values,
                  index=pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0),
                  name="cpi")
    return s.sort_index()


def load_hkma() -> pd.DataFrame:
    def rd(name):
        df = pd.read_csv(os.path.join(RAW, "hkma", name))
        df.index = pd.to_datetime(df["end_of_month"]) + pd.offsets.MonthEnd(0)
        return df
    retail = rd("hkd_retail_rates_monthly_period_average.csv")
    hibor = rd("hibor_monthly_period_average.csv")
    out = pd.DataFrame({
        "blr": retail["best_lending_rate"],
        "hibor1m": hibor["ir_1m"],
    })
    # residential mortgage survey: splice old (1996-2016) and new (2016-) vintages
    rms_n = rd("residential_mortgage_survey_monthly.csv")
    rms_o = rd("residential_mortgage_survey_monthly_old.csv")
    appr = pd.concat([
        rms_o.loc[rms_o.index < rms_n.index.min(), "new_loans_approved"],
        rms_n["new_loans_approved"]]).astype(float)
    out["mtg_approved"] = appr
    return out.sort_index()


def load_fred() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW, "fred", "FEDFUNDS.csv"))
    s = pd.Series(df["FEDFUNDS"].values,
                  index=pd.to_datetime(df["observation_date"]) + pd.offsets.MonthEnd(0),
                  name="fedfunds")
    return s.to_frame().sort_index()


def load_income_quarterly() -> pd.Series:
    """Median monthly domestic household income, one value per quarter end."""
    path = os.path.join(RAW, "censtatd",
                        "censtatd_median_household_income_quarterly.csv")
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(df["period"].astype(str).str.replace(" ", ""),
                         freq="Q").to_timestamp(how="end").normalize()
    s = pd.Series(df["median_monthly_household_income_hkd"].astype(float).values,
                  index=idx, name="med_income")
    return s[~s.index.duplicated()].sort_index()


def load_income() -> pd.Series:
    """Median income, quarterly → monthly (linear between quarter ends)."""
    return load_income_quarterly().resample("ME").interpolate("linear")


def real_time_income(lag_months: int = INCOME_LAG_MONTHS) -> pd.Series:
    """Median income as known at each month end: the latest quarter whose
    figure is treated as published (lag_months after the quarter ends),
    carried forward. Uses no later quarter, unlike the interpolated series."""
    q = load_income_quarterly()
    known = pd.Series(q.values, index=q.index + pd.offsets.MonthEnd(lag_months),
                      name="med_income_rt")
    return known.resample("ME").ffill()


def load_population() -> pd.Series:
    """Mid-year population (persons), annual → monthly (log-linear)."""
    path = os.path.join(RAW, "censtatd", "censtatd_midyear_population_annual.csv")
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["year"].astype(str)) + pd.offsets.MonthEnd(6)
    s = pd.Series(df["midyear_population"].astype(float).values, index=idx,
                  name="population").sort_index()
    return np.exp(np.log(s).resample("ME").interpolate("linear"))


def load_landreg() -> pd.DataFrame:
    path = os.path.join(RAW, "landreg", "landreg_residential_sp_monthly.csv")
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(0)
    return df.drop(columns=["month"]).sort_index()


def build_panel(start="1993-01-01") -> pd.DataFrame:
    price = parse_rvd_monthly(os.path.join(RAW, "rvd", "his_data_4.xls"), "px")
    rent = parse_rvd_monthly(os.path.join(RAW, "rvd", "his_data_3.xls"), "rt")
    cpi = load_cpi()
    hkma = load_hkma()
    fred = load_fred()
    stock_a = parse_rvd_stock(os.path.join(RAW, "rvd", "private_domestic.xls"))

    panel = pd.concat([price, rent], axis=1)
    panel["cpi"] = cpi
    panel = panel.join(hkma).join(fred)

    # annual year-end stock → monthly by log-linear interpolation, with the
    # pre-2003 years spliced onto the 2003+ definition (no village houses)
    st, splice = splice_stock(stock_a)
    st.index = pd.to_datetime(st.index.astype(str)) + pd.offsets.YearEnd(0)
    panel["stock"] = np.exp(np.log(st).resample("ME").interpolate("linear"))
    panel.attrs["stock_splice_pre2003"] = splice

    panel["med_income"] = load_income()
    panel["med_income_rt"] = real_time_income()
    panel["population"] = load_population()
    lr = load_landreg()
    panel["sp_agreements"] = lr["residential_agreements_count"]
    panel["sp_consideration"] = lr["residential_consideration_hkd_mn"]

    panel = panel.loc[panel.index >= pd.Timestamp(start)]

    # ---------------- constructed variables ----------------
    # CPI yoy inflation (%): backward-looking expected-inflation proxy
    panel["infl_yoy"] = 100.0 * (panel["cpi"] / panel["cpi"].shift(12) - 1.0)
    # real ex-post rates
    panel["real_blr"] = panel["blr"] - panel["infl_yoy"]
    panel["real_hibor"] = panel["hibor1m"] - panel["infl_yoy"]
    # logs: real price, real rent, price-rent (nominal ratio — CPI cancels)
    panel["lp_real"] = np.log(panel["px_ALL"] / panel["cpi"])
    panel["lr_real"] = np.log(panel["rt_ALL"] / panel["cpi"])
    panel["lpr"] = np.log(panel["px_ALL"] / panel["rt_ALL"])
    panel["lp_nom"] = np.log(panel["px_ALL"])
    panel["lstock"] = np.log(panel["stock"])
    panel["linc_real"] = np.log(panel["med_income"] / panel["cpi"])
    panel["linc_real_rt"] = np.log(panel["med_income_rt"] / panel["cpi"])
    panel["lvol"] = np.log(panel["sp_agreements"])
    return panel


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    panel = build_panel()
    panel.to_csv(os.path.join(OUT, "panel.csv"))
    print(panel[["px_ALL", "rt_ALL", "cpi", "blr", "lp_real", "lpr", "lstock",
                 "linc_real", "lvol"]].describe().T.to_string())
    print("\ncoverage:", panel.index.min().date(), "→", panel.index.max().date(),
          f"({len(panel)} months)")
    nn = panel.notna().sum()
    print("\nnon-null counts (key vars):")
    print(nn[["px_ALL", "rt_ALL", "cpi", "blr", "hibor1m", "med_income",
              "stock", "sp_agreements", "fedfunds"]].to_string())
