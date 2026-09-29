"""
make_tables.py — Emit the summary tables as Markdown from output/*.json|csv.
Run after run_bubbles / run_longrun / run_robustness / run_policy / run_forecast.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from pipeline import build_panel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
TABD = os.path.join(ROOT, "tables")
os.makedirs(TABD, exist_ok=True)


def w(name, text):
    with open(os.path.join(TABD, name), "w") as f:
        f.write(text)
    print(f"tables/{name}")


# T1 descriptives -------------------------------------------------------------
panel = build_panel()
vars_ = {"px_ALL": "RVD price index (1999=100)",
         "rt_ALL": "RVD rental index (1999=100)",
         "cpi": "Composite CPI",
         "med_income": "Median household income (HK$/m)",
         "blr": "Best Lending Rate (%)",
         "hibor1m": "1-month HIBOR (%)",
         "real_hibor": "Real 1m HIBOR (%)",
         "stock": "Private housing stock (units)",
         "sp_agreements": "Residential S&P agreements (/m)"}
rows = []
for v, label in vars_.items():
    s = panel[v].dropna()
    rows.append({"Variable": label, "N": len(s), "Mean": s.mean(),
                 "SD": s.std(), "Min": s.min(), "Max": s.max(),
                 "Start": str(s.index.min().date()), "End": str(s.index.max().date())})
t1 = pd.DataFrame(rows)
for c in ["Mean", "SD", "Min", "Max"]:
    t1[c] = t1[c].map(lambda x: f"{x:,.1f}")
# Keep the preformatted strings; newer tabulate versions re-parse them as numbers.
w("t1_descriptives.md", t1.to_markdown(index=False, disable_numparse=True,
                                     colalign=["left", "right"] + ["left"] * (t1.shape[1] - 2)))

# T2 unit roots ----------------------------------------------------------------
ur = pd.read_csv(os.path.join(OUTD, "unit_roots.csv"), index_col=0)
t2 = ur[["ADF_level", "ADF_level_p", "ADF_diff", "ADF_diff_p", "KPSS_level",
         "KPSS_p", "ZA_stat", "ZA_p"]].round(3)
w("t2_unit_roots.md", t2.to_markdown())

# T3 bubble tests ----------------------------------------------------------------
bb = json.load(open(os.path.join(OUTD, "bubbles.json")))
rows = []
for name, r in bb.items():
    rows.append({
        "Series": r["desc"], "T": r["T"], "SADF": round(r["sadf"], 3),
        "GSADF": round(r["gsadf"], 3),
        "CV 90/95/99": f"{r['mc_cv']['0.9']:.2f} / {r['mc_cv']['0.95']:.2f} / {r['mc_cv']['0.99']:.2f}",
        "MC p": round(r["mc_pvalue"], 3),
        "WB p": round(r["wb_pvalue"], 3),
        "Episodes (MC, 95%)": "; ".join(f"{e['start'][:7]}→{e['end'][:7]}"
                                        for e in r["episodes"]) or "—",
    })
w("t3_bubbles.md", pd.DataFrame(rows).to_markdown(index=False))

# T4 long-run --------------------------------------------------------------------
lr = json.load(open(os.path.join(OUTD, "longrun.json")))
rows = []
for sname, sp in lr["specs"].items():
    for tag in ["full", "pre2019"]:
        o = sp[tag]
        betas = ", ".join(f"{k}={v}" for k, v in o["dols_beta"].items())
        rows.append({
            "Spec": sname, "Sample": tag, "n": o["n"],
            "Bounds F": o["bounds_F"],
            "F I(1) cv95": o["bounds_cv95"][1],
            "PSS t (5% I0/I1)": f"{o['bounds_t']} ({o['bounds_t_cv95'][0]}/{o['bounds_t_cv95'][1]})",
            "I(1) p": o["bounds_p"][1],
            "DOLS β (t)": "; ".join(f"{k}: {v} ({o['dols_t'][k]})"
                                    for k, v in o["dols_beta"].items()),
            "λ (t)": f"{o['ecm_lambda']} ({o['ecm_lambda_t']})",
            "Half-life (m)": o["half_life_m"],
        })
t4 = pd.DataFrame(rows)
w("t4_longrun.md", t4.to_markdown(index=False))

# T4b Gregory-Hansen + Johansen ---------------------------------------------------
rows = []
for sname, sp in lr["specs"].items():
    for model in ["CS", "C"]:
        k = f"gh_{model}"
        if k in sp:
            rows.append({"Spec": sname, "GH model": model, "inf-ADF": sp[k]["stat"],
                         "Break": sp[k]["break"], "5% cv": sp[k]["cv05"],
                         "Reject@5%": sp[k]["reject_5pct"],
                         "Bai-Perron ECT breaks": ", ".join(sp.get("bp_breaks", []))
                         + (" (capped at 5)" if sp.get("bp_capped") else "")})
w("t4b_gh.md", pd.DataFrame(rows).to_markdown(index=False))
jo = pd.read_csv(os.path.join(OUTD, "johansen.csv"), index_col=0).round(2)
w("t4c_johansen.md", jo.to_markdown())

# T5 policy ------------------------------------------------------------------------
po = json.load(open(os.path.join(OUTD, "policy.json")))
rows = []
for dep in ["price", "volume"]:
    for c, v in po[f"its_{dep}"].items():
        rows.append({"Dep.": f"Δlog {dep}", "Dummy": c, "Coef": v["b"],
                     "t (HAC)": v["t"], "p": v["p"]})
w("t5_its.md", pd.DataFrame(rows).to_markdown(index=False))

lp_rows = []
for shock in ["tightening", "ease"]:
    for dep in ["price", "volume"]:
        irf = pd.read_csv(os.path.join(OUTD, f"lp_{shock}_{dep}.csv"), index_col=0)
        for h in [0, 4, 8, 12, 24]:
            if h in irf.index:
                r = irf.loc[h]
                sig = "*" if r["p"] <= 0.05 else ""
                lp_rows.append({"Shock": shock, "Response": f"cum. log {dep}",
                                "h": h, "β_h": round(r["beta"], 3),
                                "placebo p": round(r["p"], 3),
                                "95% CI (placebo)": f"[{r['lo95']:.3f}, {r['hi95']:.3f}]{sig}"})
w("t5b_lp.md", pd.DataFrame(lp_rows).to_markdown(index=False))

# T6 forecast -----------------------------------------------------------------------
summ = pd.read_csv(os.path.join(OUTD, "oos_summary.csv")).round(4)
w("t6_oos.md", summ.to_markdown(index=False))

# T7 robustness ----------------------------------------------------------------------
rb = json.load(open(os.path.join(OUTD, "robustness.json")))
w("t7_robustness.md", "```json\n" + json.dumps(rb, indent=2) + "\n```")
print("done")
