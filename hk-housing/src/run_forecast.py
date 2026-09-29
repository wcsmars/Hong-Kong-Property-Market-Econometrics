"""
run_forecast.py — Pseudo out-of-sample forecast comparison across models.

Target: log real price index (S2 fundamentals: real income, real BLR).
Income enters as known at each date (pipeline.real_time_income: the latest
published quarter, carried forward), so no origin uses a later quarter.
Rolling origins from 60% of the sample, step 2 months, horizons 1/3/12.
Outputs: output/oos_h{1,3,12}.csv, oos_summary.csv
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import forecast
from pipeline import build_panel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

panel = build_panel()
y = panel["lp_real"]
X = panel[["linc_real_rt", "real_blr"]]

oos = forecast.rolling_oos(y, X, start_frac=0.6, horizons=(1, 3, 12), step=2)
for h, df in oos.items():
    df.to_csv(os.path.join(OUTD, f"oos_h{h}.csv"))

summ = forecast.oos_summary(oos, step=2)
summ.to_csv(os.path.join(OUTD, "oos_summary.csv"), index=False)
print(summ.round(4).to_string(index=False))
