"""
policy.py — Causal-ish evaluation of Hong Kong's demand-side cooling measures.

Two complementary designs:

1. Interrupted time series (ITS): monthly log-return (and log-volume)
   regressions with post-measure level/step dummies and short pulse windows,
   controls for macro fundamentals, HAC (Newey-West) inference. Identifies
   the *average* shift in dynamics after each measure conditional on
   fundamentals — credible to the extent measures are exogenous to the
   month-t shock (they respond to *past* appreciation, addressed by
   conditioning on lagged returns).

2. Jordà (2005) local projections: dynamic response of cumulative returns
   and volumes h = 0..H months after a measure dummy switches on,
       y_{t+h} - y_{t-1} = α_h + β_h·D_t + Γ_h'W_t + ε_{t+h},
   with W_t lagged returns + macro controls. D_t is 1 in only one month (the
   2024 withdrawal) or four (the pooled tightenings), so Newey-West errors are
   not valid: the single event is fitted exactly and contributes nothing to
   the variance estimate. Inference is by randomization over placebo event
   dates instead (see local_projection); HAC errors are kept as a diagnostic.

The measure set and dates (cross-checked against published announcements and prior studies):
   SSD  2010-11-20  Special Stamp Duty (resale within 24m taxed 5-15%)
   BSD  2012-10-27  Buyer's Stamp Duty (15% on non-PR buyers) + SSD enhanced
   DSD  2013-02-23  Doubled ad-valorem stamp duty (all > HK$2m)
   NRSD 2016-11-05  Flat 15% AVD on second+ homes
   EASE 2024-02-28  Full withdrawal of SSD/BSD/NRSD ("spicy measures" end)
Each measure is dated to the month-end of its effective date, so the base
month t-1 is wholly pre-measure and h = 0 is the (partly treated) adoption
month; h = 0 responses are not read as policy effects. The 25 October 2023
partial relaxation (BSD and NRSD halved, SSD holding period shortened) is not
modelled separately, so the 2024 withdrawal is the second step of a two-stage
relaxation. Land Registry volume counts agreements when lodged for
registration, up to about a month after signing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MEASURES = {
    "SSD": "2010-11-30",
    "BSD": "2012-10-31",
    "DSD": "2013-02-28",
    "NRSD": "2016-11-30",
    "EASE": "2024-02-29",
}
MEASURE_DATES = [pd.Timestamp(d) for d in MEASURES.values()]
# placebo event dates stay away from every measure, including the unmodelled
# October 2023 partial relaxation
PLACEBO_EXCLUDE = MEASURE_DATES + [pd.Timestamp("2023-10-31")]


def build_policy_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Step dummies (1 from measure month onward) and 6-month pulse dummies."""
    out = pd.DataFrame(index=index)
    for name, date in MEASURES.items():
        d = pd.Timestamp(date)
        out[f"{name}_step"] = (index >= d).astype(float)
        out[f"{name}_pulse6"] = ((index >= d) &
                                 (index < d + pd.DateOffset(months=6))).astype(float)
    return out


def its_regression(dep: pd.Series, controls: pd.DataFrame, dummies: pd.DataFrame,
                   dummy_cols: list[str], n_ar: int = 3, hac_lags: int = 6):
    """ITS with AR terms of the dependent variable + macro controls + dummies."""
    parts = {f"{dep.name}_l{l}": dep.shift(l) for l in range(1, n_ar + 1)}
    Z = pd.concat([pd.DataFrame(parts), controls, dummies[dummy_cols]], axis=1)
    aligned = pd.concat([dep, Z], axis=1).dropna()
    res = sm.OLS(aligned.iloc[:, 0], sm.add_constant(aligned.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return res


def placebo_calendars(index: pd.DatetimeIndex, events: list, exclude: list,
                      window: int = 24) -> list:
    """Placebo event calendars for randomization inference.

    Every calendar shifts the real event months by a common offset, so a
    single event becomes each eligible month and a pooled set keeps its
    spacing. A calendar is eligible when all its months are in `index` and
    more than `window` months from every month in `exclude` (all measures).
    Returns a list of integer-position arrays into `index`.
    """
    months = index.year * 12 + index.month
    pos = {m: i for i, m in enumerate(months)}
    ev = np.array([d.year * 12 + d.month for d in events])
    ex = np.array([d.year * 12 + d.month for d in exclude])
    out = []
    for offset in range(int(months.min() - ev.min()), int(months.max() - ev.max()) + 1):
        shifted = ev + offset
        if offset == 0 or any(m not in pos for m in shifted):
            continue
        if np.min(np.abs(shifted[:, None] - ex[None, :])) <= window:
            continue
        out.append(np.array([pos[m] for m in shifted]))
    return out


def random_calendars(index: pd.DatetimeIndex, n_events: int, exclude: list,
                     window: int = 24, n_draws: int = 4999, seed: int = 0) -> list:
    """Placebo calendars of n_events distinct months drawn independently from
    the eligible months (more than `window` months from every month in
    `exclude`). Unlike placebo_calendars this ignores the real events'
    spacing, so pooled calendars can also use periods too short to hold a
    shifted copy of the whole event calendar."""
    months = index.year * 12 + index.month
    ex = np.array([d.year * 12 + d.month for d in exclude])
    eligible = np.flatnonzero(np.min(np.abs(months.to_numpy()[:, None] - ex[None, :]),
                                     axis=1) > window)
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(eligible, n_events, replace=False)) for _ in range(n_draws)]


def placebo_betas(W: np.ndarray, yv: np.ndarray, calendars: list) -> np.ndarray:
    """Coefficient on a placebo dummy (1 on the rows of calendar c) in the OLS
    of yv on [W, dummy], for every calendar, by Frisch-Waugh-Lovell:
    β_c = d'My / d'Md with M the annihilator of W."""
    Q, _ = np.linalg.qr(W)
    ey = yv - Q @ (Q.T @ yv)
    return np.array([ey[c].sum() / (len(c) - np.sum(Q[c].sum(axis=0) ** 2))
                     for c in calendars])


def local_projection(y: pd.Series, shock: pd.Series, controls: pd.DataFrame,
                     H: int = 24, n_ar: int = 3, exclude_window: int = 24,
                     scheme: str = "shift", n_draws: int = 4999, seed: int = 0):
    """Jordà LP IRF of cumulative Δy on a policy step-change indicator.

    shock should be the first difference of a step dummy (=1 in the
    adoption month or months). Inference is by randomization: the shock is
    moved to placebo calendars away from every measure (scheme "shift":
    every shift of the real calendar, placebo_calendars; scheme "random":
    n_draws independent sets of months, random_calendars), the real event
    months stay in the regression as controls, and β is re-estimated by
    Frisch-Waugh-Lovell. p = (1 + #{|β_placebo| ≥ |β|}) / (1 + n). The
    90/95% bands are β ± q, with q the order statistic of |β_placebo| that
    inverts that test, so a band excludes 0 exactly when p <= 0.10 or 0.05. se_hac is the Newey-West (lag h+1) standard error, kept only as a
    diagnostic because it is not valid with 1-4 events. Horizons stop once
    an event month has no outcome in the sample.
    Returns DataFrame indexed by h with beta, se_hac, p, lo90, hi90, lo95,
    hi95 and n_placebo.
    """
    dy = y.diff()
    events = list(shock.index[shock == 1])
    rows = []
    for h in range(H + 1):
        lhs = (y.shift(-h) - y.shift(1)).rename("cum")
        parts = {"shock": shock}
        for l in range(1, n_ar + 1):
            parts[f"dy_l{l}"] = dy.shift(l)
        Z = pd.concat([pd.DataFrame(parts), controls], axis=1)
        aligned = pd.concat([lhs, Z], axis=1).dropna()
        if len(aligned) < 30 or not all(e in aligned.index for e in events):
            break
        res = sm.OLS(aligned["cum"], sm.add_constant(aligned.drop(columns="cum"))
                     ).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        b = float(res.params["shock"])
        # the real shock stays among the controls W of every placebo regression
        W = sm.add_constant(aligned.drop(columns="cum")).to_numpy(float)
        if scheme == "shift":
            cals = placebo_calendars(aligned.index, events, PLACEBO_EXCLUDE, exclude_window)
        else:
            cals = random_calendars(aligned.index, len(events), PLACEBO_EXCLUDE,
                                    exclude_window, n_draws, seed + h)
        placebo = placebo_betas(W, aligned["cum"].to_numpy(float), cals)
        n = len(placebo)
        p = float((1 + np.sum(np.abs(placebo) >= abs(b))) / (1 + n))
        ordered = np.sort(np.abs(placebo))
        half = {}
        for level in (0.90, 0.95):
            k = int(np.ceil(level * (n + 1)))       # |β| > half[level]  <=>  p <= 1 - level
            half[level] = ordered[k - 1] if k <= n else np.inf
        if h == 0:
            used = aligned.index[np.concatenate(cals)]
            span = (str(used.min().date()), str(used.max().date()))
        rows.append({"h": h, "beta": b, "se_hac": float(res.bse["shock"]), "p": p,
                     "lo90": b - half[0.90], "hi90": b + half[0.90],
                     "lo95": b - half[0.95], "hi95": b + half[0.95],
                     "n_placebo": n})
    out = pd.DataFrame(rows).set_index("h")
    out.attrs["placebo_span"] = span   # first and last placebo month at h = 0
    return out
