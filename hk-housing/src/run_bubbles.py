"""
run_bubbles.py — Recursive explosiveness tests on HK residential prices.

Series tested (1993-01 … latest, monthly):
  lp_nom : log nominal RVD all-class price index
  lp_real: log real price index (CPI-deflated)
  lpr    : log price-rent ratio  ← the present-value-theory object: under no
           bubble and I(1) fundamentals, p_t - r_t cannot be explosive
           (Campbell-Shiller; Phillips et al. apply GSADF to exactly this)

For each series: SADF, GSADF, Monte-Carlo critical values and p-value (exact
T, iid random-walk null, 49,999 reps so the pointwise date-stamping sequence
is stable), date-stamped episodes vs the 95% BSADF critical-value sequence
with the ceil(log T) minimum-duration rule, and a wild bootstrap (4,999 reps)
whose null keeps the serial correlation (Phillips-Shi autoregressive null)
and the changing volatility (time-aligned residuals) of monthly price
changes; the 1997-98 crisis and 2003 SARS trough make homoskedasticity
untenable here. Episodes against the bootstrap sequence (episodes_wb) are
pointwise, not corrected for testing many endpoints.

Lag order: k = 1 in every ADF regression, observed and simulated (BIC
selects one lag for all three series); the bootstrap null autoregression
also has one lag.
Results → output/bubbles.json, output/bsadf_<series>.csv
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import psy
from pipeline import build_panel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

panel = build_panel()
SERIES = {"lp_nom": "log nominal price index",
          "lp_real": "log real price index",
          "lpr": "log price-rent ratio"}

results = {}
K_EMP = 1          # lag order of every ADF regression, observed and simulated
MC_REPS = 49_999
WB_REPS = 4_999
null_cache = {}    # the MC null depends only on (T, w0), shared across series

for name, desc in SERIES.items():
    y = panel[name].dropna()
    T = len(y)
    w0 = psy.min_window(T)
    print(f"\n=== {name} ({desc}); T={T}, w0={w0} ===")

    seq = psy.bsadf(y.values, k=K_EMP, w0=w0)
    g = float(np.nanmax(seq[np.isfinite(seq)]))
    s = psy.sadf(y.values, k=K_EMP, w0=w0)

    if (T, w0) not in null_cache:
        null_cache[(T, w0)] = psy.mc_critical_values(
            T, w0=w0, n_reps=MC_REPS, seed=20260612, k=K_EMP)
    cv = null_cache[(T, w0)]
    mc_p = float((np.sum(cv["null_gsadf"] >= g) + 1) / (MC_REPS + 1))
    wb = psy.wild_bootstrap_gsadf(y.values, k=K_EMP, w0=w0, n_reps=WB_REPS, seed=31)

    episodes = psy.date_stamp(seq, cv["bsadf_cv_seq"][0.95], y.index)
    episodes_wb = psy.date_stamp(seq, wb["bsadf_cv_seq"][0.95], y.index)

    print(f"SADF={s:.3f}  GSADF={g:.3f}")
    print(f"MC CV (90/95/99): {cv['gsadf_cv'][0.90]:.3f} / "
          f"{cv['gsadf_cv'][0.95]:.3f} / {cv['gsadf_cv'][0.99]:.3f}")
    print(f"MC p(GSADF) = {mc_p:.4f}; wild-bootstrap p(GSADF) = {wb['pvalue']:.4f} "
          f"(null AR coef {np.round(wb['psi'], 3).tolist()})")
    for ep in episodes:
        print(f"  episode: {ep['start'].date()} → {ep['end'].date()} "
              f"({ep['n_periods']}m, peak {ep['peak'].date()}, stat {ep['peak_stat']:.2f})")

    pd.DataFrame({
        "bsadf": seq,
        "cv90": cv["bsadf_cv_seq"][0.90], "cv95": cv["bsadf_cv_seq"][0.95],
        "cv99": cv["bsadf_cv_seq"][0.99],
        "cv95_wb": wb["bsadf_cv_seq"][0.95],
        "y": y.values,
    }, index=y.index).to_csv(os.path.join(OUTD, f"bsadf_{name}.csv"))

    results[name] = {
        "desc": desc, "T": T, "w0": int(w0), "k": K_EMP,
        "sadf": s, "gsadf": g,
        "mc_cv": {str(q): v for q, v in cv["gsadf_cv"].items()},
        "mc_pvalue": mc_p, "mc_reps": MC_REPS,
        "wb_pvalue": wb["pvalue"], "wb_reps": WB_REPS,
        "wb_cv": {str(q): v for q, v in wb["gsadf_cv"].items()},
        "wb_null_ar": wb["psi"],
        "episodes": [{k2: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                      for k2, v in ep.items()} for ep in episodes],
        "episodes_wb": [{k2: (str(v.date()) if isinstance(v, pd.Timestamp) else v)
                         for k2, v in ep.items()} for ep in episodes_wb],
    }

with open(os.path.join(OUTD, "bubbles.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nsaved output/bubbles.json")
