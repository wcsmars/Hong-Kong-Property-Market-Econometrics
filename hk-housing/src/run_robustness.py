"""
run_robustness.py — Sensitivity analysis for aggregation, lags and segments.

R1. Quarterly re-aggregation: income is quarterly (interpolated to monthly in
    the baseline) and stock annual — re-run ARDL bounds for S2/S3 on quarterly
    averages, where no within-quarter interpolation is doing any work.
R2. Lag-order sensitivity: GSADF with k = 0 (baseline k = 1). BIC selects
    one lag, so the zero-lag statistic is judged against a wild bootstrap
    whose null keeps that serial correlation; the iid critical value is
    reported for reference only.
R3. Segment heterogeneity: GSADF + date-stamping on class A (small flats)
    and class E (luxury) price-rent ratios.
R4. HIBOR-based real rate in S3 (baseline: BLR).
R5-R6. Same-sample and alternative rate measurements for S3.
R7. Drop the provisional months (Feb-Apr 2026).
R8. Zero-lag price-rent dating against the serial-correlation-robust
    bootstrap sequence.
Monte-Carlo nulls use the lag order of the tested statistic (49,999 reps);
bootstraps use 4,999 reps.
Outputs: output/robustness.json
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller

import psy
import longrun
from pipeline import build_panel

MC_REPS = 49_999
WB_REPS = 4_999
_null_cache = {}


def mc_null(T, w0, k, seed):
    """Monte-Carlo null for GSADF/BSADF with k lags, cached by (T, w0, k, seed)."""
    key = (T, w0, k, seed)
    if key not in _null_cache:
        _null_cache[key] = psy.mc_critical_values(T, w0=w0, n_reps=MC_REPS,
                                                  seed=seed, k=k)
    return _null_cache[key]

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)
res = {}

panel = build_panel()
panel["lsk_pc"] = panel["lstock"] - np.log(panel["population"])

# R1 ------------------------------------------------------- quarterly bounds
q = panel[["lp_real", "lpr", "linc_real", "real_blr", "real_hibor"]].resample("QE").mean()
for sname, (yv, xs) in {"S2q": ("lp_real", ["linc_real", "real_blr"]),
                        "S3q": ("lpr", ["real_blr"]),
                        "S3Hq": ("lpr", ["real_hibor"])}.items():
    d = q[[yv] + xs].dropna()
    b = longrun.ardl_bounds(d[yv], d[xs], maxlag=4)
    res[sname] = {"n": len(d), "bounds_F": round(b["bounds_stat"], 3),
                  "cv95": [round(float(b["bounds_crit"].loc[95.0, "lower"]), 2),
                           round(float(b["bounds_crit"].loc[95.0, "upper"]), 2)]}
    print(f"{sname}: F={res[sname]['bounds_F']} cv95={res[sname]['cv95']} n={len(d)}")

# R2 --------------------------------------------------------------- k = 0 GSADF
for name in ["lp_nom", "lp_real", "lpr"]:
    y = panel[name].dropna()
    w0 = psy.min_window(len(y))
    bic_lag = int(adfuller(y.values, maxlag=12, regression="c", autolag="BIC")[2])
    g0 = psy.gsadf(y.values, k=0, w0=w0)
    cv = mc_null(len(y), w0, 0, 99)
    wb = psy.wild_bootstrap_gsadf(y.values, k=0, w0=w0, n_reps=WB_REPS, seed=31,
                                  ar_lags=bic_lag)
    res[f"gsadf_k0_{name}"] = {"gsadf": round(g0, 3), "bic_lag": bic_lag,
                               "cv95_iid": round(cv["gsadf_cv"][0.95], 3),
                               "wb_pvalue_ar_null": round(wb["pvalue"], 3)}
    print(f"GSADF k=0 {name}: {g0:.3f} (iid cv95 {cv['gsadf_cv'][0.95]:.3f}; "
          f"BIC lag {bic_lag}; AR-null wild-bootstrap p {wb['pvalue']:.3f})")

# R3 ------------------------------------------------- class A / E price-rent
for cls in ["A", "E"]:
    s = np.log(panel[f"px_{cls}"] / panel[f"rt_{cls}"]).dropna()
    w0 = psy.min_window(len(s))
    seq = psy.bsadf(s.values, k=1, w0=w0)
    g = float(np.nanmax(seq[np.isfinite(seq)]))
    cv = mc_null(len(s), w0, 1, 77)
    wb = psy.wild_bootstrap_gsadf(s.values, k=1, w0=w0, n_reps=WB_REPS, seed=31)
    eps = psy.date_stamp(seq, cv["bsadf_cv_seq"][0.95], s.index)
    res[f"lpr_class{cls}"] = {
        "gsadf": round(g, 3), "cv95": round(cv["gsadf_cv"][0.95], 3),
        "reject5": bool(g > cv["gsadf_cv"][0.95]),
        "wb_pvalue": round(wb["pvalue"], 3),
        "episodes": [f"{e['start'].date()}→{e['end'].date()}" for e in eps]}
    print(f"P/R class {cls}: GSADF={g:.3f} (cv95 {cv['gsadf_cv'][0.95]:.3f}); "
          f"episodes: {res[f'lpr_class{cls}']['episodes']}")

# R4 -------------------------------------------------------- HIBOR real rate
d = panel[["lpr", "real_hibor"]].dropna()
b = longrun.ardl_bounds(d["lpr"], d[["real_hibor"]], maxlag=6)
dd = longrun.dols(d["lpr"], d[["real_hibor"]])
res["S3_hibor"] = {"n": len(d), "bounds_F": round(b["bounds_stat"], 3),
                   "cv95": [round(float(b["bounds_crit"].loc[95.0, "lower"]), 2),
                            round(float(b["bounds_crit"].loc[95.0, "upper"]), 2)],
                   "dols_beta": dd["beta"].round(4).to_dict()}
print(f"S3 with real HIBOR: F={res['S3_hibor']['bounds_F']} "
      f"beta={res['S3_hibor']['dols_beta']}")

# R5 -------------------------- same-sample rate decomposition (BLR vs HIBOR)
d5 = panel[["lpr", "real_blr", "real_hibor"]].dropna()
b_blr = longrun.ardl_bounds(d5["lpr"], d5[["real_blr"]], maxlag=6)
res["S3_blr_samesample"] = {"n": len(d5),
                            "bounds_F": round(b_blr["bounds_stat"], 3)}
print(f"S3-BLR on the HIBOR sample (n={len(d5)}): F={b_blr['bounds_stat']:.3f}")

# R6 ----------------------- alternative rate measurements for the S3H anchor
eom = pd.read_csv(os.path.join(RAW := os.path.join(ROOT, "data", "raw"),
                               "hkma", "hibor_monthly_end_period.csv"))
eom.index = pd.to_datetime(eom["end_of_month"]) + pd.offsets.MonthEnd(0)
panel["real_hibor_eom"] = eom["ir_1m"] - panel["infl_yoy"]
d6 = panel[["lpr", "real_hibor_eom"]].dropna()
b6 = longrun.ardl_bounds(d6["lpr"], d6[["real_hibor_eom"]], maxlag=6)
res["S3H_eom_fixing"] = {"n": len(d6), "bounds_F": round(b6["bounds_stat"], 3)}
print(f"S3H with end-of-month fixing: F={b6['bounds_stat']:.3f} (n={len(d6)})")

comp = pd.read_csv(os.path.join(RAW, "hkma", "composite_interest_rate_monthly.csv"))
comp.index = pd.to_datetime(comp["end_of_month"]) + pd.offsets.MonthEnd(0)
panel["real_comp"] = comp["interest_rate"] - panel["infl_yoy"]
d6b = panel[["lpr", "real_comp"]].dropna()
b6b = longrun.ardl_bounds(d6b["lpr"], d6b[["real_comp"]], maxlag=6)
res["S3_composite_rate"] = {"n": len(d6b), "bounds_F": round(b6b["bounds_stat"], 3),
                            "cv95": [round(float(b6b["bounds_crit"].loc[95.0, "lower"]), 2),
                                     round(float(b6b["bounds_crit"].loc[95.0, "upper"]), 2)]}
print(f"S3 with composite funding rate (2004+): F={b6b['bounds_stat']:.3f} (n={len(d6b)})")

# R7 ------------------- drop provisional months (Feb-Apr 2026): re-run cores
trunc = panel.loc[:"2026-01-31"]
for name in ["lp_nom", "lp_real", "lpr"]:
    yv = trunc[name].dropna()
    w0t = psy.min_window(len(yv))
    gt = psy.gsadf(yv.values, k=1, w0=w0t)
    cvt = mc_null(len(yv), w0t, 1, 20260612)
    wbt = psy.wild_bootstrap_gsadf(yv.values, k=1, w0=w0t, n_reps=WB_REPS, seed=31)
    res[f"trunc_gsadf_{name}"] = {"gsadf": round(gt, 3),
                                  "cv95": round(cvt["gsadf_cv"][0.95], 3),
                                  "reject5": bool(gt > cvt["gsadf_cv"][0.95]),
                                  "wb_pvalue": round(wbt["pvalue"], 3)}
d7 = trunc[["lpr", "real_hibor"]].dropna()
b7 = longrun.ardl_bounds(d7["lpr"], d7[["real_hibor"]], maxlag=6)
res["trunc_S3H"] = {"n": len(d7), "bounds_F": round(b7["bounds_stat"], 3)}
print(f"ex-provisional: GSADF {[res[f'trunc_gsadf_{n}']['gsadf'] for n in ['lp_nom','lp_real','lpr']]}, "
      f"S3H F={b7['bounds_stat']:.3f}")

# R8 ------------------------------- k = 0 price-rent dating + wild bootstrap
# dated against the bootstrap sequence whose null keeps the BIC-selected
# serial correlation, not the iid sequence the zero-lag statistic oversizes
y8 = panel["lpr"].dropna()
w08 = psy.min_window(len(y8))
lag8 = res["gsadf_k0_lpr"]["bic_lag"]
seq8 = psy.bsadf(y8.values, k=0, w0=w08)
wb8 = psy.wild_bootstrap_gsadf(y8.values, k=0, w0=w08, n_reps=WB_REPS, seed=31,
                               ar_lags=lag8)
eps8 = psy.date_stamp(seq8, wb8["bsadf_cv_seq"][0.95], y8.index)
peak8 = y8.index[int(np.nanargmax(np.where(np.isfinite(seq8), seq8, -np.inf)))]
res["lpr_k0_dating"] = {
    "gsadf": round(float(np.nanmax(seq8[np.isfinite(seq8)])), 3),
    "sup_date": str(peak8.date()), "wb_pvalue": round(wb8["pvalue"], 3),
    "episodes": [f"{e['start'].date()}→{e['end'].date()}" for e in eps8]}
print(f"lpr k=0: episodes {res['lpr_k0_dating']['episodes']}, "
      f"sup {res['lpr_k0_dating']['sup_date']}, wb p={res['lpr_k0_dating']['wb_pvalue']}")

with open(os.path.join(OUTD, "robustness.json"), "w") as f:
    json.dump(res, f, indent=2)
print("\nsaved output/robustness.json")
