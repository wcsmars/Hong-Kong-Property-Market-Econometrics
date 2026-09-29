"""
run_longrun.py — Long-run equilibrium between real prices and fundamentals.

Three theory-grounded specifications (logs, monthly):
  S1 inverted demand:  lp_real ~ linc_real + real_blr + lsk_pc
                       (lsk_pc = log stock per capita: supply relative to
                        demographic demand)
  S2 parsimonious:     lp_real ~ linc_real + real_blr
  S3 user cost:        lpr ~ real_blr   (Poterba / HMS: log(P/R) ≈ −log(uc))

For each: ARDL bounds (full sample AND pre-2019 subsample), DOLS, ECM,
Gregory-Hansen cointegration with endogenous regime shift, Bai-Perron breaks
on the equilibrium error. A Johansen test on (lp_real, linc_real, lsk_pc) uses
two lagged differences, with the trace statistic also saved for 0-12 lags and
the lag orders AIC, BIC and HQIC select. The "pre2019" subsample ends in
December 2019. Outputs: output/longrun.json, output/johansen.csv,
output/equilibrium_path_<spec>.csv, output/unit_roots.csv.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import select_order

import longrun
from pipeline import build_panel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

panel = build_panel()
panel["lsk_pc"] = panel["lstock"] - np.log(panel["population"])

SPECS = {
    "S1": ("lp_real", ["linc_real", "real_blr", "lsk_pc"]),
    "S2": ("lp_real", ["linc_real", "real_blr"]),
    "S3": ("lpr", ["real_blr"]),
    "S3H": ("lpr", ["real_hibor"]),
}
PRE_END = "2019-12-31"   # "pre2019" subsample: through December 2019

df = panel[["lp_real", "lpr", "linc_real", "real_blr", "lsk_pc", "lstock"]].dropna()
print(f"core sample: {df.index.min().date()} → {df.index.max().date()} (n={len(df)})")

res = {"sample": [str(df.index.min().date()), str(df.index.max().date())],
       "n": len(df), "specs": {}}

# unit roots ------------------------------------------------------------------
ur = longrun.unit_root_table(df, ["lp_real", "lpr", "linc_real", "real_blr",
                                  "lsk_pc"])
ur.to_csv(os.path.join(OUTD, "unit_roots.csv"))
print("\n--- unit roots (ADF level p / diff p / KPSS p / ZA p) ---")
print(ur[["ADF_level_p", "ADF_diff_p", "KPSS_p", "ZA_p"]].round(3).to_string())

# Johansen on the demand block (price, income, stock per capita) -------------
# Two lagged differences is a fixed choice, not an information-criterion one,
# and the rank-0 rejection is lag-sensitive, so the trace statistic is also
# saved for every order from 0 to 12 lags with the IC-selected orders.
JO_VARS = ["lp_real", "linc_real", "lsk_pc"]
JO_LAGS = 2
jo_tab, jo_beta = longrun.johansen(df[JO_VARS], det_order=0, k_ar_diff=JO_LAGS)
jo_tab.to_csv(os.path.join(OUTD, "johansen.csv"))
res["johansen"] = jo_tab.reset_index().to_dict("records")
res["johansen_beta"] = jo_beta.round(4).to_dict()
res["johansen_spec"] = {"variables": JO_VARS, "det_order": 0,
                        "deterministic": "unrestricted constant",
                        "k_ar_diff": JO_LAGS}
jo_sel = select_order(df[JO_VARS], maxlags=12, deterministic="co")
res["johansen_lag_selection"] = {"aic": int(jo_sel.aic), "bic": int(jo_sel.bic),
                                 "hqic": int(jo_sel.hqic)}
lag_rows = []
for lags in range(0, 13):
    t_l, b_l = longrun.johansen(df[JO_VARS], det_order=0, k_ar_diff=lags)
    lag_rows.append({"k_ar_diff": lags, "trace_r0": round(float(t_l.loc[0, "trace"]), 3),
                     "trace_cv95": round(float(t_l.loc[0, "trace_cv95"]), 3),
                     "trace_reject": bool(t_l.loc[0, "trace_reject"]),
                     "income_coef": round(float(-b_l["linc_real"]), 3)})
res["johansen_lag_sensitivity"] = lag_rows
print("Johansen trace r=0 by lagged differences: "
      + ", ".join(f"{r['k_ar_diff']}:{r['trace_r0']:.1f}" for r in lag_rows)
      + f" (cv95 {lag_rows[0]['trace_cv95']:.1f}); IC choices {res['johansen_lag_selection']}")

for sname, (yv, xs) in SPECS.items():
    out = {"y": yv, "X": xs}
    sdf = panel[[yv] + xs].dropna()        # per-spec sample (HIBOR starts 1996-07)
    y, X = sdf[yv], sdf[xs]
    y_pre, X_pre = y.loc[:PRE_END], X.loc[:PRE_END]

    for tag, (yy, XX) in {"full": (y, X), "pre2019": (y_pre, X_pre)}.items():
        b = longrun.ardl_bounds(yy, XX, maxlag=6)
        d = longrun.dols(yy, XX, leads_lags=4, hac_lags=6)
        ec = longrun.ecm(yy, XX, d["lr_resid"], n_lags=2)
        out[tag] = {
            "n": int(len(yy.dropna())),
            "bounds_F": round(b["bounds_stat"], 3),
            "bounds_t": round(b["bounds_t"], 3),
            "bounds_t_cv95": list(longrun.PSS_T_BOUNDS_5PCT[b["k"]]),
            "bounds_k": b["k"],
            "bounds_cv95": [round(float(b["bounds_crit"].loc[95.0, "lower"]), 3),
                            round(float(b["bounds_crit"].loc[95.0, "upper"]), 3)],
            "bounds_cv90": [round(float(b["bounds_crit"].loc[90.0, "lower"]), 3),
                            round(float(b["bounds_crit"].loc[90.0, "upper"]), 3)],
            "bounds_p": [round(float(b["bounds_pvals"]["lower"]), 4),
                         round(float(b["bounds_pvals"]["upper"]), 4)],
            "ardl_order": [int(v) for v in np.atleast_1d(b["ardl_order"])],
            "dols_beta": d["beta"].round(4).to_dict(),
            "dols_t": (d["beta"] / d["se"]).round(2).to_dict(),
            "ecm_lambda": round(ec["lambda"], 4),
            "ecm_lambda_t": round(ec["lambda_t"], 2),
            "half_life_m": (round(ec["half_life_months"], 1)
                            if ec["half_life_months"] is not None else None),
        }
        if tag == "full":
            # first column is the spec's dependent variable (lp_real or lpr)
            eq = pd.DataFrame({yv: y, "fitted": y - d["lr_resid"],
                               "ect": d["lr_resid"]})
            eq.to_csv(os.path.join(OUTD, f"equilibrium_path_{sname}.csv"))
            bp = longrun.bai_perron_breaks(d["lr_resid"], max_breaks=5,
                                           min_size=36)
            out["bp_breaks"] = [str(dd.date()) for dd in bp["break_dates"]]
            out["bp_capped"] = bool(bp["capped"])

    # Gregory-Hansen with regime shift (and level shift for S3)
    for model in (["CS", "C"] if sname != "S3" else ["C", "CS"]):
        gh = longrun.gregory_hansen(y, X, model=model)
        out[f"gh_{model}"] = {
            "stat": round(gh["stat"], 3),
            "break": str(gh["break_date"].date()),
            "cv05": gh["cv"][0.05], "cv10": gh["cv"][0.10],
            "reject_5pct": bool(gh["reject_5pct"]),
        }

    res["specs"][sname] = out
    print(f"\n=== {sname}: {yv} ~ {' + '.join(xs)} ===")
    for tag in ["full", "pre2019"]:
        o = out[tag]
        print(f" [{tag}] n={o['n']} boundsF={o['bounds_F']} t={o['bounds_t']} "
              f"(cv95 {o['bounds_cv95']}, p {o['bounds_p']}) | DOLS {o['dols_beta']} "
              f"| λ={o['ecm_lambda']} (t={o['ecm_lambda_t']}), HL={o['half_life_m']}m")
    for model in ["CS", "C"]:
        k = f"gh_{model}"
        if k in out:
            print(f" [GH-{model}] stat={out[k]['stat']} break={out[k]['break']} "
                  f"cv05={out[k]['cv05']} reject5%={out[k]['reject_5pct']}")
    if "bp_breaks" in out:
        print(f" [Bai-Perron ECT breaks] {out['bp_breaks']}")

with open(os.path.join(OUTD, "longrun.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print("\nsaved output/longrun.json")
