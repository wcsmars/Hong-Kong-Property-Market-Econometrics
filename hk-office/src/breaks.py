"""
breaks.py — Structural change and the decentralisation gradient.

(a) Bai-Perron multiple mean-shift breaks (hand-rolled dynamic programme,
    BIC-selected number of breaks, 15% trimming) on:
      - monthly Grade A real rent growth (1993-02+),
      - the Central premium: log(avg Grade A rent Central / KwunTong),
        2013-02+ (Kwun Tong enters the district panel then).
(b) Markov-switching mean-variance model (Hamilton 1989) on monthly Grade A
    rent growth, with a seeded start-parameter search so reruns reproduce
    it; smoothed regime probabilities saved for the figure.
(c) District convergence (Grade A, monthly average rents 1999+, 7 districts):
      - sigma-convergence: cross-sectional std of the non-Central districts'
        log rents relative to Central, over all observed districts and over
        balanced panels that hold the district set fixed (Kwun Tong enters in
        2013-02; Mong Kok, Sheung Wan and Quarry Bay have gaps). The pre-2019
        vs 2019+ change in mean dispersion is tested by a HAC regression on a
        2019+ dummy; Levene's test compares only its variability;
      - beta-convergence: Δ ln(rent_d / rent_Central) on lagged level;
      - pairwise Engle-Granger cointegration of each district with Central,
        on the longest run of consecutive months after filling interior gaps
        of up to three months.

Outputs: output/breaks.json, output/ms_probs.csv, output/sigma_path.csv,
         output/breaks_log.txt
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller, coint

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
LOG: list[str] = []
DISTRICTS = ["SheungWan", "Central", "CausewayBay", "QuarryBay", "TST",
             "MongKok", "KwunTong"]


def log(msg: str):
    LOG.append(str(msg))
    print(msg)


# ------------------------------------------------------------- Bai-Perron
def bai_perron_mean(y: np.ndarray, max_breaks: int = 5, trim: float = 0.15):
    """Global SSR-minimising mean-shift segmentation via dynamic programming.
    Number of breaks selected by BIC. Returns (n_breaks, break_idx, bic_path).
    """
    T = len(y)
    h = max(int(np.floor(trim * T)), 2)
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y ** 2)])

    def seg_cost(i, j):  # SSR of segment y[i:j] (j exclusive)
        n = j - i
        s, s2 = c1[j] - c1[i], c2[j] - c2[i]
        return s2 - s * s / n

    # cost[i][j]: minimal SSR for y[0:j] with i breaks before j
    INF = np.inf
    cost = np.full((max_breaks + 1, T + 1), INF)
    prev = np.full((max_breaks + 1, T + 1), -1, dtype=int)
    for j in range(h, T + 1):
        cost[0, j] = seg_cost(0, j)
    for b in range(1, max_breaks + 1):
        for j in range((b + 1) * h, T + 1):
            for k in range(b * h, j - h + 1):
                c = cost[b - 1, k] + seg_cost(k, j)
                if c < cost[b, j]:
                    cost[b, j], prev[b, j] = c, k
    res = {}
    for b in range(max_breaks + 1):
        ssr = cost[b, T]
        if not np.isfinite(ssr):
            continue
        k_params = 2 * b + 1  # b breaks, b+1 means, (variance common)
        bic = T * np.log(ssr / T) + k_params * np.log(T)
        # backtrack
        bks, j, bb = [], T, b
        while bb > 0:
            j = prev[bb, j]
            bks.append(j)
            bb -= 1
        res[b] = {"ssr": float(ssr), "bic": float(bic),
                  "breaks_idx": sorted(bks)}
    n_best = min(res, key=lambda b: res[b]["bic"])
    return n_best, res[n_best]["breaks_idx"], {b: r["bic"] for b, r in res.items()}


def run_bai_perron(series: pd.Series, name: str, max_breaks: int = 5) -> dict:
    s = series.dropna()
    n, idx, bics = bai_perron_mean(s.values, max_breaks=max_breaks)
    dates = [str(s.index[i])[:10] for i in idx]
    seg_means = []
    bounds = [0] + idx + [len(s)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg_means.append(float(s.values[a:b].mean()))
    log(f"Bai-Perron [{name}]: {n} breaks at {dates}; "
        f"segment means {[round(m, 4) for m in seg_means]}")
    return {"series": name, "n_breaks": int(n), "break_dates": dates,
            "segment_means": seg_means, "bic_path": bics, "n_obs": int(len(s))}


# -------------------------------------------------------- Markov switching
def markov(panel: pd.DataFrame, seed: int = 0) -> dict:
    y = (100 * panel["lr_real_A"].diff()).dropna()
    mod = MarkovRegression(y.values, k_regimes=2, trend="c",
                           switching_variance=True)
    # statsmodels draws the random start-parameter search from numpy's
    # global RNG; seed it so reruns reproduce the fit, then restore it.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        res = mod.fit(search_reps=20)
    finally:
        np.random.set_state(state)
    pr = pd.DataFrame({"p_regime0": res.smoothed_marginal_probabilities[:, 0]},
                      index=y.index)
    pr.to_csv(os.path.join(OUT, "ms_probs.csv"))
    p00 = float(res.params[0]); p10 = float(res.params[1])
    mu0, mu1 = float(res.params[2]), float(res.params[3])
    s0, s1 = float(np.sqrt(res.params[4])), float(np.sqrt(res.params[5]))
    dur0 = 1.0 / (1.0 - p00) if p00 < 1 else np.inf
    p11 = 1.0 - p10
    dur1 = 1.0 / (1.0 - p11) if p11 < 1 else np.inf
    out = {
        "mu_regime0_pct_m": mu0, "mu_regime1_pct_m": mu1,
        "sigma_regime0": s0, "sigma_regime1": s1,
        "p_stay0": p00, "p_stay1": p11,
        "exp_duration0_m": dur0, "exp_duration1_m": dur1,
        "llf": float(res.llf), "n": int(len(y)),
    }
    log(f"Markov-switching: regime0 μ={mu0:.3f}%/m σ={s0:.2f} (dur {dur0:.0f}m); "
        f"regime1 μ={mu1:.3f}%/m σ={s1:.2f} (dur {dur1:.0f}m)")
    return out


# ------------------------------------------------------------ convergence
def sigma_split(rel: pd.DataFrame, hac_lags: int = 12) -> tuple[pd.Series, dict]:
    """Cross-sectional std of log rent relatives, pre-2019 vs 2019+.

    `rel` holds each non-Central district's log rent minus Central's, so the
    std runs over those relatives ("districts" lists the panel, Central
    included; "relatives" the columns the std uses). It runs over whichever
    columns are observed in a month, so pass a balanced frame (only months
    with every district observed) to hold the district set fixed.

    The change in mean dispersion is tested by regressing sigma_t on a 2019+
    dummy with Newey-West (HAC) errors, because sigma_t is highly
    autocorrelated. Levene's p-value tests only whether sigma_t is equally
    variable in the two periods.
    """
    sigma = rel.std(axis=1).dropna()
    post_dummy = (sigma.index >= "2019-01-01").astype(float)
    pre = sigma[post_dummy == 0]
    post = sigma[post_dummy == 1]
    lev = stats.levene(pre, post)
    hac = sm.OLS(sigma.values, sm.add_constant(post_dummy)).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return sigma, {
        "districts": ["Central"] + [c.removeprefix("ar_A_") for c in rel.columns],
        "relatives": [c.removeprefix("ar_A_") for c in rel.columns],
        "start": str(sigma.index[0].date()), "end": str(sigma.index[-1].date()),
        "n_pre2019": int(len(pre)), "n_post2019": int(len(post)),
        "pre2019_mean": float(pre.mean()), "post2019_mean": float(post.mean()),
        "mean_change": float(hac.params[1]), "mean_change_hac_t": float(hac.tvalues[1]),
        "mean_change_hac_p": float(hac.pvalues[1]), "hac_lags": hac_lags,
        "levene_p": float(lev.pvalue),
    }


def fill_short_gaps(s: pd.Series, limit: int = 3) -> pd.Series:
    """Linearly fill interior gaps of at most `limit` consecutive months."""
    missing = s.isna()
    run = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(missing & (run_len <= limit)), filled)


def longest_complete_run(frame: pd.DataFrame) -> pd.DataFrame:
    """Longest stretch of consecutive rows with every column observed."""
    ok = frame.notna().all(axis=1)
    run = (ok != ok.shift()).cumsum()
    sizes = ok[ok].groupby(run[ok]).size()
    rows = ok[ok][run[ok] == sizes.idxmax()].index
    return frame.loc[rows[0]:rows[-1]]


def convergence(dist: pd.DataFrame) -> dict:
    cols = {d: f"ar_A_{d}" for d in DISTRICTS}
    R = dist[[c for c in cols.values() if c in dist.columns]].copy()
    R = R.dropna(how="all")
    lr = np.log(R)
    rel = lr.sub(lr["ar_A_Central"], axis=0)
    rel = rel.drop(columns=["ar_A_Central"])

    # sigma-convergence: cross-sectional std of log relatives. Kwun Tong
    # enters in 2013-02 about 1.1 log points below Central and Mong Kok has
    # gaps, so the all-observed std also moves when the district set changes;
    # the balanced panels keep the set fixed.
    frames = {
        "all_observed": rel,
        "balanced": rel.dropna(how="any"),
        "balanced_ex_kwuntong": rel.drop(columns=["ar_A_KwunTong"]).dropna(how="any"),
    }
    sig, paths = {}, {}
    for name, frame in frames.items():
        paths[f"sigma_{name}"], s = sigma_split(frame)
        sig[f"sigma_{name}"] = s
        log(f"sigma-convergence [{name}, {s['start'][:7]}+, {len(s['districts'])} districts, "
            f"{len(s['relatives'])} relatives to Central]: "
            f"mean pre-2019={s['pre2019_mean']:.4f}, post={s['post2019_mean']:.4f} "
            f"(HAC p={s['mean_change_hac_p']:.4f}; Levene p={s['levene_p']:.4f})")
    pd.DataFrame(paths).rename_axis(rel.index.name).to_csv(
        os.path.join(OUT, "sigma_path.csv"))

    # beta-convergence: pooled Δrel on lagged rel (monthly)
    panels = []
    for c in rel.columns:
        panels.append(pd.DataFrame({"drel": rel[c].diff(), "lag": rel[c].shift(1)}))
    pool = pd.concat(panels).dropna()
    beta = sm.OLS(pool["drel"], sm.add_constant(pool["lag"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 12})
    hl = float(np.log(0.5) / np.log(1 + beta.params["lag"])) \
        if -1 < beta.params["lag"] < 0 else np.nan
    log(f"beta-convergence: β={beta.params['lag']:.4f} "
        f"(t={beta.tvalues['lag']:.2f}), half-life={hl:.0f}m")

    # pairwise Engle-Granger with Central + Central premium trend. The test's
    # lags assume consecutive months, so short gaps are filled and the test
    # runs on the longest complete stretch rather than across longer holes.
    pairs = {}
    for d in DISTRICTS:
        if d == "Central" or f"ar_A_{d}" not in lr.columns:
            continue
        raw = lr[[f"ar_A_{d}", "ar_A_Central"]]
        sub = longest_complete_run(raw.apply(fill_short_gaps))
        t, p, _ = coint(sub[f"ar_A_{d}"], sub["ar_A_Central"])
        pairs[d] = {"eg_t": float(t), "eg_p": float(p), "n": int(len(sub)),
                    "start": str(sub.index[0].date()), "end": str(sub.index[-1].date()),
                    "months_filled": int(raw.loc[sub.index].isna().any(axis=1).sum())}
    log("pairwise EG vs Central: " +
        ", ".join(f"{d}: p={v['eg_p']:.3f}" for d, v in pairs.items()))

    # Central premium over KwunTong (Kowloon East proxy)
    prem = (lr["ar_A_Central"] - lr["ar_A_KwunTong"]).dropna().rename("premium")
    adf_p = adfuller(prem.values)[1]
    log(f"Central/KwunTong premium: mean={prem.mean():.3f} "
        f"start={prem.iloc[0]:.3f} end={prem.iloc[-1]:.3f} ADF p={adf_p:.3f}")
    return {
        **sig,
        "beta": float(beta.params["lag"]), "beta_t": float(beta.tvalues["lag"]),
        "beta_half_life_m": hl,
        "pairwise_eg": pairs,
        "premium_mean": float(prem.mean()),
        "premium_start": float(prem.iloc[0]), "premium_end": float(prem.iloc[-1]),
        "premium_adf_p": float(adf_p),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = pd.read_csv(os.path.join(PRO, "monthly_panel.csv"),
                        index_col=0, parse_dates=True)
    dist = pd.read_csv(os.path.join(PRO, "district_monthly.csv"),
                       index_col=0, parse_dates=True)
    results = {}
    results["bp_rent_growth"] = run_bai_perron(
        panel["lr_real_A"].diff(), "d_lr_real_A (monthly)")
    lrC = np.log(dist["ar_A_Central"]) - np.log(dist["ar_A_KwunTong"])
    results["bp_central_premium"] = run_bai_perron(lrC, "central_premium")
    results["markov"] = markov(panel)
    results["convergence"] = convergence(dist)
    with open(os.path.join(OUT, "breaks.json"), "w") as f:
        clean = json.loads(json.dumps(results, default=float),
                           parse_constant=lambda _: None)
        json.dump(clean, f, indent=2, allow_nan=False)
    with open(os.path.join(OUT, "breaks_log.txt"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
