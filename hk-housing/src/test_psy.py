"""
test_psy.py — Validation of the PSY implementation before it touches real data.

1. EXACTNESS: windowed ADF statistic must match statsmodels.adfuller
   (regression='c', fixed lag) to ~1e-8 on identical samples, for k = 0
   and k > 0, full-sample and subsample windows, including the k = 0 and
   k = 1 prefix-sum paths used inside the simulations.
2. POWER: a series with a known embedded explosive episode
   (y_t = 1.04·y_{t-1} + ε_t for t ∈ [120, 160)) must be flagged by GSADF
   at the 95% MC critical value, and date-stamping must locate the episode
   within a few periods of the truth.
3. SIZE: under H0 (pure random walk), the rejection rate of GSADF at the
   95% MC critical value across 200 null draws must lie in [0.013, 0.097]
   (binomial 99% band around 0.05 ≈ [0.013, 0.097]).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from statsmodels.tsa.stattools import adfuller

import psy

rng = np.random.default_rng(123)
failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- exactness
y = np.cumsum(rng.standard_normal(300))

for k in (0, 1, 4):
    ours = psy.adf_stat(y, k=k)
    sm_stat = adfuller(y, maxlag=k, regression="c", autolag=None)[0]
    check(f"ADF exactness k={k} (full sample)", abs(ours - sm_stat) < 1e-8,
          f"ours={ours:.10f} sm={sm_stat:.10f}")

# subsample window [50, 199]
for k in (0, 2):
    ours = psy.adf_stat(y, k=k, s=50, e=199)
    sm_stat = adfuller(y[50:200], maxlag=k, regression="c", autolag=None)[0]
    check(f"ADF exactness k={k} (window [50,199])", abs(ours - sm_stat) < 1e-8,
          f"ours={ours:.10f} sm={sm_stat:.10f}")

# k=0 fast path vs k-lag general path must agree at k=0
fast = psy.adf_stat(y, k=0, s=30, e=250)
gen = psy._adf_window_klags(y, 30, 250, 0)
check("k=0 fast path == general path", abs(fast - gen) < 1e-9,
      f"fast={fast:.12f} gen={gen:.12f}")

# k=1 prefix-sum path (used by the Monte-Carlo nulls and bootstraps) must
# agree with the general path and statsmodels
P1 = psy._prefix_sums_k1(y)
fast1 = psy._adf_window_k1(P1, 30, 250)
gen1 = psy._adf_window_klags(y, 30, 250, 1)
sm1 = adfuller(y[30:251], maxlag=1, regression="c", autolag=None)[0]
check("k=1 fast path == general path == statsmodels",
      max(abs(fast1 - gen1), abs(fast1 - sm1)) < 1e-9,
      f"fast={fast1:.12f} gen={gen1:.12f} sm={sm1:.12f}")

# ---------------------------------------------------------------- power
# PSY-style DGP: random walk with a non-trivial initial level (y_0 = 50, as
# in PSY's simulations where the level matters for the explosive signal),
# switching to rho = 1.04 during t in [120, 160).
T = 320
eps = rng.standard_normal(T)
yb = np.zeros(T)
yb[0] = 50.0
for t in range(1, T):
    if t == 160:                       # collapse back to pre-bubble level
        yb[t] = yb[119] + eps[t]
    else:
        rho = 1.04 if 120 <= t < 160 else 1.0
        yb[t] = rho * yb[t - 1] + eps[t]

w0 = psy.min_window(T)
cv = psy.mc_critical_values(T, w0=w0, n_reps=499, seed=99)
stat = psy.gsadf(yb, k=0, w0=w0)
check("GSADF power (explosive episode rejected at 95%)",
      stat > cv["gsadf_cv"][0.95],
      f"stat={stat:.3f} cv95={cv['gsadf_cv'][0.95]:.3f}")

seq = psy.bsadf(yb, k=0, w0=w0)
eps_idx = np.arange(T)
episodes = psy.date_stamp(seq, cv["bsadf_cv_seq"][0.95], eps_idx)
found = any(120 <= ep["start"] <= 145 and 150 <= ep["end"] <= 175
            for ep in episodes)
check("Date-stamping locates episode (~[120,160))", found,
      f"episodes={[(ep['start'], ep['end']) for ep in episodes]}")

# ---------------------------------------------------------------- size
n_null = 200
rej = 0
null_rng = np.random.default_rng(2024)
for i in range(n_null):
    yn = np.cumsum(null_rng.standard_normal(T))
    if psy.gsadf(yn, k=0, w0=w0) > cv["gsadf_cv"][0.95]:
        rej += 1
rate = rej / n_null
check("GSADF size under H0 within binomial band", 0.013 <= rate <= 0.097,
      f"rejection rate={rate:.3f} (nominal 0.05)")

# ---------------------------------------------------------------- bootstrap sanity
wb = psy.wild_bootstrap_gsadf(yb, w0=w0, n_reps=199, seed=5)
check("Wild bootstrap rejects explosive series (p < 0.05)", wb["pvalue"] < 0.05,
      f"p={wb['pvalue']:.4f}")
yn = np.cumsum(np.random.default_rng(77).standard_normal(T))
wb0 = psy.wild_bootstrap_gsadf(yn, w0=w0, n_reps=199, seed=5)
check("Wild bootstrap does not reject a random walk (p > 0.10)",
      wb0["pvalue"] > 0.10, f"p={wb0['pvalue']:.4f}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PSY VALIDATION TESTS PASSED")
