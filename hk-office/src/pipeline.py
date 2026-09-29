"""
pipeline.py — Build estimation panels for the Hong Kong office study.

Sources (original snapshot dated 2026-06-13; required inputs are listed in
input_manifest.json at the repository root):
  RVD his_data_8.xls   : private office RENTAL indices by grade
                         (monthly 1993+, quarterly 1978+, annual 1981+)
  RVD his_data_9.xls   : private office PRICE indices by grade
                         (monthly 1993+, quarterly 1986+, annual 1981+)
  RVD his_data_10.xls  : Grade A office rent indices, core districts
                         (monthly 1993+, quarterly 1989+, annual 1984+)
  RVD his_data_6.xls   : average RENTS  $/m2/month, grade x district
                         (monthly sheet, 1999+)
  RVD his_data_7.xls   : average PRICES $/m2,       grade x district
                         (monthly sheet, 1999+)
  RVD private_office.xls : completions / stock / vacancy / take-up by grade,
                         annual 1985+ (vacancy includes rates)
  HKMA API             : 1m HIBOR (1996-07+), BLR (1980+), composite rate
  C&SD                 : composite CPI monthly (1980-10+)
  FRED                 : GS10, FEDFUNDS (monthly)
  World Bank API       : HK real GDP, constant LCU, annual 1961-2024

Outputs (data/processed/):
  monthly_panel.csv      one row per month, grade-level indices + macro + yields
  quarterly_indices.csv  long-span grade indices (rent 1978+, price 1986+)
  annual_stockflow.csv   stock-flow + vacancy + annual indices + GDP
  district_monthly.csv   Grade A district rents/yields + his_data_10 indices

Conventions follow hk-housing/src/pipeline.py: hard validation gates
(calendar uniqueness, gap-freeness, range checks) raise rather than warn.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")

# his_data_8/9 monthly & quarterly sheets: grade header anchored at these
# columns; the printed value lands in anchor or anchor+1 (merged cells).
GRADE_COLS = {"A": (7, 8), "B": (10, 11), "C": (13, 14), "ALL": (16, 17)}

# his_data_6/7 monthly: grade blocks anchored at A=7, B=28, C=49; within a
# block, district headers sit at anchor + offset and data at header + 1.
DISTRICT_OFFSETS = {
    "SheungWan": 0, "Central": 3, "CausewayBay": 6, "QuarryBay": 9,
    "TST": 12, "MongKok": 15, "KwunTong": 18,
}
GRADE_ANCHORS = {"A": 7, "B": 28, "C": 49}

# his_data_10 monthly: header col -> (name, kind); data col = header + 2
D10_SERIES = {
    7: ("d10_rent_SWC", "rent: Sheung Wan / Central"),
    12: ("d10_rent_WCCB", "rent: Wan Chai / Causeway Bay"),
    17: ("d10_rent_TST", "rent: Tsim Sha Tsui"),
    22: ("d10_price_CORE", "price: core districts"),
}


def _coerce_num(v):
    """RVD cell -> float. Handles '(58.1)' (<20 transactions), '*', '-', blanks,
    and percentages stored as text ('9.5%' -> 0.095, as numeric rate cells)."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace("(", "").replace(")", "").replace("*", "").strip()
    if s in ("", "-", "N.A.", "n.a.", "@", "#"):
        return np.nan
    scale = 1.0
    if s.endswith("%"):
        s, scale = s[:-1].strip(), 0.01
    try:
        return float(s) * scale
    except ValueError:
        return np.nan


def _coerce_year(v):
    if pd.isna(v):
        return None
    try:
        y = float(str(v).strip())
    except ValueError:
        return None
    return int(y) if 1970 <= y <= 2100 else None


def _validate_monthly(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.index.duplicated().any():
        raise ValueError(f"{name}: duplicated months {df.index[df.index.duplicated()][:4]}")
    gaps = pd.date_range(df.index.min(), df.index.max(), freq="ME").difference(df.index)
    if len(gaps):
        raise ValueError(f"{name}: missing months {gaps[:4]}")
    return df


def _first_data_col(df: pd.DataFrame, lo: int, hi: int, min_count: int = 30) -> int | None:
    """Column in [lo, hi] with the most coercible numeric cells (>= min_count)."""
    best, best_n = None, 0
    for c in range(lo, min(hi + 1, df.shape[1])):
        n = df[c].map(_coerce_num).notna().sum()
        if n > best_n:
            best, best_n = c, n
    return best if best_n >= min_count else None


# ---------------------------------------------------------------- his_data_8/9
def parse_grade_monthly(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None)
    year, rows = None, []
    for _, r in df.iterrows():
        y = _coerce_year(r.iloc[1])
        if y is not None:
            year = y
        m = _coerce_num(r.iloc[5])
        if np.isnan(m) if isinstance(m, float) else m is None:
            continue
        m = int(m)
        if not 1 <= m <= 12 or year is None:
            continue
        rec = {"date": pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)}
        for g, cols in GRADE_COLS.items():
            val = np.nan
            for c in cols:
                val = _coerce_num(r.iloc[c]) if c < len(r) else np.nan
                if not np.isnan(val):
                    break
            rec[f"{prefix}_{g}"] = val
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("date").sort_index()
    # trim leading/trailing all-NaN rows but keep internal NaNs: RVD publishes
    # '^' (no index, <5 transactions) in thin months — these are real data
    # features (transaction droughts), not parse failures.
    valid = out.notna().any(axis=1)
    out = out.loc[valid[valid].index.min(): valid[valid].index.max()]
    n_gap = out[f"{prefix}_ALL"].isna().sum()
    if n_gap > 12:
        raise ValueError(f"{path}: too many unpublished months ({n_gap})")
    return _validate_monthly(out, path)


def parse_grade_quarterly(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Quarterly  按季", header=None)
    year, rows = None, []
    for _, r in df.iterrows():
        y = _coerce_year(r.iloc[1])
        if y is not None:
            year = y
        m_start = _coerce_num(r.iloc[3])
        if np.isnan(m_start) or year is None or m_start not in (1, 4, 7, 10):
            continue
        q = int(m_start) // 3 + 1
        rec = {"quarter": pd.Period(f"{year}Q{q}")}
        for g, cols in GRADE_COLS.items():
            val = np.nan
            for c in cols:
                val = _coerce_num(r.iloc[c]) if c < len(r) else np.nan
                if not np.isnan(val):
                    break
            rec[f"{prefix}_{g}"] = val
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("quarter").sort_index()
    out = out.dropna(how="all")
    if out.index.duplicated().any():
        raise ValueError(f"{path} quarterly: duplicated quarters")
    return out


def parse_grade_annual(path: str, prefix: str) -> pd.DataFrame:
    """Annual sheet: year at col 2; grade headers detected from 'Grade X' row."""
    df = pd.read_excel(path, sheet_name="Annual  按年", header=None)
    hdr_row = None
    for i in range(min(12, len(df))):
        vals = [str(v) for v in df.iloc[i].tolist()]
        if any("Grade A" in v for v in vals):
            hdr_row = i
            break
    if hdr_row is None:
        raise ValueError(f"{path}: no 'Grade A' header row in annual sheet")
    anchors = {}
    for c in range(df.shape[1]):
        v = str(df.iat[hdr_row, c])
        for g in ("A", "B", "C"):
            if v.strip() == f"Grade {g}":
                anchors[g] = c
        if "Overall" in v:
            anchors["ALL"] = c
    recs = {}
    for _, r in df.iterrows():
        y = _coerce_year(r.iloc[2])
        if y is None:
            continue
        rec = {}
        for g, h in anchors.items():
            val = np.nan
            for c in range(h, min(h + 3, len(r))):
                val = _coerce_num(r.iloc[c])
                if not np.isnan(val):
                    break
            rec[f"{prefix}_{g}"] = val
        recs[y] = rec
    out = pd.DataFrame(recs).T.sort_index()
    out.index.name = "year"
    return out.dropna(how="all")


# ---------------------------------------------------------------- his_data_10
def parse_d10_monthly(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=None)
    colmap = {}
    for h, (name, _) in D10_SERIES.items():
        c = _first_data_col(df, h, h + 4)
        if c is None:
            raise ValueError(f"{path}: no data column for {name}")
        colmap[name] = c
    year, rows = None, []
    for _, r in df.iterrows():
        y = _coerce_year(r.iloc[1])
        if y is not None:
            year = y
        m = _coerce_num(r.iloc[5])
        if np.isnan(m) or year is None or not 1 <= m <= 12:
            continue
        rec = {"date": pd.Timestamp(year, int(m), 1) + pd.offsets.MonthEnd(0)}
        for name, c in colmap.items():
            rec[name] = _coerce_num(r.iloc[c])
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("date").sort_index().dropna(how="all")
    return _validate_monthly(out, path)


# ---------------------------------------------------------------- his_data_6/7
def _detect_grade_anchors(df: pd.DataFrame) -> dict[str, int]:
    """Row whose cells are exactly 'A','B','C' marks the grade-block anchors.
    his_data_6 and his_data_7 have different block positions (70 vs 76 cols)."""
    for i in range(min(10, len(df))):
        cells = {str(df.iat[i, c]).strip(): c for c in range(df.shape[1])}
        if {"A", "B", "C"} <= set(cells):
            return {g: cells[g] for g in ("A", "B", "C")}
    raise ValueError("grade anchor row not found")


def parse_avg_monthly(path: str, prefix: str) -> pd.DataFrame:
    """Average rents ($/m2/month) or prices ($/m2), grade x district, monthly 1999+."""
    df = pd.read_excel(path, sheet_name="Monthly  按月", header=None)
    anchors = _detect_grade_anchors(df)
    colmap = {}
    for g, anchor in anchors.items():
        for dist, off in DISTRICT_OFFSETS.items():
            h = anchor + off
            c = _first_data_col(df, h, h + 2, min_count=24)
            if c is not None:
                colmap[f"{prefix}_{g}_{dist}"] = c
    year, rows = None, []
    for _, r in df.iterrows():
        y = _coerce_year(r.iloc[1])
        if y is not None:
            year = y
        m = _coerce_num(r.iloc[5])
        if np.isnan(m) or year is None or not 1 <= m <= 12:
            continue
        rec = {"date": pd.Timestamp(year, int(m), 1) + pd.offsets.MonthEnd(0)}
        for name, c in colmap.items():
            rec[name] = _coerce_num(r.iloc[c])
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("date").sort_index().dropna(how="all")
    return _validate_monthly(out, path)


# ------------------------------------------------------------ private_office
def parse_stockflow(path: str) -> pd.DataFrame:
    sheets = {
        "Completions_落成量": ("comp", False),
        "Stock_總存量": ("stock", False),
        "Vacancy_空置量": ("vac", True),
        "Take-Up_使用量": ("takeup", False),
    }
    grades = ["A", "B", "C", "TOT"]
    out = {}
    for sheet, (name, has_rate) in sheets.items():
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        recs = {}
        for _, r in df.iterrows():
            y = _coerce_year(r.iloc[2])
            if y is None:
                continue
            rec = {}
            if has_rate:
                cols = [(4, 5), (6, 7), (8, 9), (10, 11)]
                for g, (ca, cr) in zip(grades, cols):
                    rec[f"{name}_{g}"] = _coerce_num(r.iloc[ca])
                    rec[f"{name}rate_{g}"] = _coerce_num(r.iloc[cr])
            else:
                for g, ca in zip(grades, [4, 6, 8, 10]):
                    rec[f"{name}_{g}"] = _coerce_num(r.iloc[ca])
            recs[y] = rec
        out[name] = pd.DataFrame(recs).T
    res = pd.concat(out.values(), axis=1).sort_index()
    res.index.name = "year"
    # range gates
    vr = res[[c for c in res.columns if c.startswith("vacrate")]]
    if (vr.dropna() < 0).any().any() or (vr.dropna() > 0.6).any().any():
        raise ValueError("vacancy rate outside [0, 0.6]")
    # a printed rate must exist and agree with vacancy / stock (rates are
    # rounded to 0.1pp), so an unparsed or misplaced cell cannot slip through
    for g in grades:
        implied = res[f"vac_{g}"] / res[f"stock_{g}"]
        rate = res[f"vacrate_{g}"]
        missing = rate.isna() & implied.notna()
        if missing.any():
            raise ValueError(f"vacrate_{g} missing for {list(res.index[missing])}")
        if ((rate - implied).abs() > 0.001).any():
            raise ValueError(f"vacrate_{g} disagrees with vac_{g} / stock_{g}")
    return res


# ---------------------------------------------------------------------- macro
def load_cpi() -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, "censtatd", "censtatd_cpi_composite_monthly.csv"))
    return pd.Series(df["cpi_composite_index"].values,
                     index=pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0),
                     name="cpi").sort_index()


def load_rates() -> pd.DataFrame:
    def rd(name):
        df = pd.read_csv(os.path.join(RAW, "hkma", name))
        df.index = pd.to_datetime(df["end_of_month"]) + pd.offsets.MonthEnd(0)
        return df
    hibor = rd("hibor_monthly_period_average.csv")
    retail = rd("hkd_retail_rates_monthly_period_average.csv")
    out = pd.DataFrame({
        "hibor1m": hibor["ir_1m"], "hibor3m": hibor["ir_3m"],
        "blr": retail["best_lending_rate"],
    })
    comp = pd.read_csv(os.path.join(RAW, "hkma", "composite_interest_rate_monthly.csv"))
    comp.index = pd.to_datetime(comp["end_of_month"]) + pd.offsets.MonthEnd(0)
    out["composite_rate"] = comp["interest_rate"]
    gs10 = pd.read_csv(os.path.join(RAW, "fred", "GS10.csv"))
    out["gs10"] = pd.Series(gs10["GS10"].values,
                            index=pd.to_datetime(gs10["observation_date"]) + pd.offsets.MonthEnd(0))
    ff = pd.read_csv(os.path.join(RAW, "fred", "FEDFUNDS.csv"))
    out["fedfunds"] = pd.Series(ff["FEDFUNDS"].values,
                                index=pd.to_datetime(ff["observation_date"]) + pd.offsets.MonthEnd(0))
    return out.sort_index()


def load_gdp() -> pd.Series:
    df = pd.read_csv(os.path.join(RAW, "worldbank", "hkg_gdp_constant_lcu.csv"))
    s = pd.Series(df["gdp_constant_lcu"].values, index=df["year"].astype(int),
                  name="gdp_real")
    s.index.name = "year"
    return s.sort_index()


# --------------------------------------------------------------------- build
def build():
    os.makedirs(OUT, exist_ok=True)

    rent_m = parse_grade_monthly(os.path.join(RAW, "rvd", "his_data_8.xls"), "rt")
    price_m = parse_grade_monthly(os.path.join(RAW, "rvd", "his_data_9.xls"), "px")
    rent_q = parse_grade_quarterly(os.path.join(RAW, "rvd", "his_data_8.xls"), "rt")
    price_q = parse_grade_quarterly(os.path.join(RAW, "rvd", "his_data_9.xls"), "px")
    rent_a = parse_grade_annual(os.path.join(RAW, "rvd", "his_data_8.xls"), "rt")
    price_a = parse_grade_annual(os.path.join(RAW, "rvd", "his_data_9.xls"), "px")
    d10 = parse_d10_monthly(os.path.join(RAW, "rvd", "his_data_10.xls"))
    avg_rent_m = parse_avg_monthly(os.path.join(RAW, "rvd", "his_data_6.xls"), "ar")
    avg_price_m = parse_avg_monthly(os.path.join(RAW, "rvd", "his_data_7.xls"), "ap")
    sf = parse_stockflow(os.path.join(RAW, "rvd", "private_office.xls"))
    cpi = load_cpi()
    rates = load_rates()
    gdp = load_gdp()

    # ------------------------------------------------ monthly panel (1993+)
    panel = pd.concat([rent_m, price_m], axis=1).join(d10, how="left")
    panel["cpi"] = cpi
    panel = panel.join(rates, how="left")
    panel["infl_yoy"] = 100.0 * (panel["cpi"] / panel["cpi"].shift(12) - 1.0)
    panel["real_hibor"] = panel["hibor1m"] - panel["infl_yoy"]
    panel["real_gs10"] = panel["gs10"] - panel["infl_yoy"]
    panel["real_blr"] = panel["blr"] - panel["infl_yoy"]
    for g in ("A", "B", "C", "ALL"):
        panel[f"lr_{g}"] = np.log(panel[f"rt_{g}"])
        panel[f"lp_{g}"] = np.log(panel[f"px_{g}"])
        panel[f"lpr_{g}"] = panel[f"lp_{g}"] - panel[f"lr_{g}"]
        panel[f"lr_real_{g}"] = np.log(panel[f"rt_{g}"] / panel["cpi"])
        panel[f"lp_real_{g}"] = np.log(panel[f"px_{g}"] / panel["cpi"])

    # direct yields (12 x monthly rent / price), by grade x district, 1999+
    ylds = {}
    for g in ("A", "B", "C"):
        cols = []
        for dist in DISTRICT_OFFSETS:
            rc, pc = f"ar_{g}_{dist}", f"ap_{g}_{dist}"
            if rc in avg_rent_m.columns and pc in avg_price_m.columns:
                y = 12.0 * avg_rent_m[rc] / avg_price_m[pc]
                ylds[f"yld_{g}_{dist}"] = y
                cols.append(f"yld_{g}_{dist}")
        if cols:
            ylds[f"yld_{g}_mean"] = pd.DataFrame({c: ylds[c] for c in cols}).mean(axis=1)
    ylds = pd.DataFrame(ylds)
    bad = ylds.stack().pipe(lambda s: s[(s < 0.005) | (s > 0.30)])
    if len(bad):
        raise ValueError(f"implausible yields:\n{bad.head()}")
    panel = panel.join(ylds[[c for c in ylds.columns if c.endswith("_mean")]],
                       how="left")
    panel = panel.loc[panel.index >= "1993-01-31"]
    _validate_monthly(panel, "monthly_panel")
    panel.to_csv(os.path.join(OUT, "monthly_panel.csv"))

    # -------------------------------------------- quarterly indices (1978+)
    q = pd.concat([rent_q, price_q], axis=1).sort_index()
    cpi_q = cpi.groupby(pd.PeriodIndex(cpi.index, freq="Q")).mean().rename("cpi")
    q = q.join(cpi_q, how="left")
    q.to_csv(os.path.join(OUT, "quarterly_indices.csv"))

    # --------------------------------------------- annual stock-flow (1981+)
    ann = pd.concat([sf,
                     rent_a.add_prefix("idx_"), price_a.add_prefix("idx_")],
                    axis=1).sort_index()
    ann = ann.join(gdp, how="left")
    cpi_a = cpi.groupby(cpi.index.year).mean().rename("cpi")
    cpi_a.index.name = "year"
    ann = ann.join(cpi_a, how="left")
    hibor_a = rates["hibor1m"].groupby(rates.index.year).mean().rename("hibor1m")
    blr_a = rates["blr"].groupby(rates.index.year).mean().rename("blr")
    ann = ann.join(hibor_a, how="left").join(blr_a, how="left")
    # stock identity: Δstock = completions − demolitions ± reclassification.
    # Residuals are real (demolitions; RVD regrading 2015-17) — report, and
    # hard-gate only on internal consistency: grades must sum to the total.
    for g in ("A", "B", "C", "TOT"):
        ann[f"dstock_{g}"] = ann[f"stock_{g}"].diff()
    ann["stock_residual"] = ann["dstock_TOT"] - ann["comp_TOT"]
    gsum = ann[["stock_A", "stock_B", "stock_C"]].sum(axis=1)
    rel = ((gsum - ann["stock_TOT"]).abs() / ann["stock_TOT"]).dropna()
    if (rel > 0.01).any():
        raise ValueError(f"grade stocks do not sum to total: {rel[rel > 0.01].head()}")
    ann.to_csv(os.path.join(OUT, "annual_stockflow.csv"))

    # ------------------------------------------------- district monthly panel
    dist = d10.join(avg_rent_m, how="outer").join(avg_price_m, how="outer")
    dist = dist.join(pd.DataFrame(ylds), how="left")
    dist.to_csv(os.path.join(OUT, "district_monthly.csv"))

    return panel, q, ann, dist


if __name__ == "__main__":
    panel, q, ann, dist = build()
    print("monthly_panel:", panel.index.min().date(), "→", panel.index.max().date(),
          f"({len(panel)} months, {panel.shape[1]} cols)")
    print("quarterly_indices:", q.index.min(), "→", q.index.max(), f"({len(q)} quarters)")
    print("annual_stockflow:", ann.index.min(), "→", ann.index.max(), f"({len(ann)} years)")
    print("district_monthly:", dist.index.min().date(), "→", dist.index.max().date(),
          f"({dist.shape[1]} cols)")
    print("\nkey coverage (non-null):")
    keys = ["rt_A", "px_A", "rt_ALL", "px_ALL", "hibor1m", "cpi", "yld_A_mean",
            "d10_rent_SWC"]
    print(panel[keys].notna().sum().to_string())
    print("\nvacancy rate (Grade A, last 6):")
    print(ann["vacrate_A"].dropna().tail(6).to_string())
    print("\nyield means (last obs):", {c: round(panel[c].dropna().iloc[-1], 4)
          for c in panel.columns if c.startswith("yld_")})
