"""
psy.py — Recursive right-tailed unit-root tests for explosive behaviour.

Implements, from first principles:

  * ADF statistic on an arbitrary subsample window [s, e] with k lagged
    differences (fitted intercept, no trend) — the regression of
    Phillips, Shi & Yu (2015, Int. Econ. Rev.), eq. (1):

        Δy_t = α + β·y_{t-1} + Σ_{i=1..k} ψ_i Δy_{t-i} + ε_t,
        ADF = β̂ / se(β̂).

  * SADF  (Phillips, Wu & Yu, 2011): sup over forward-expanding windows
    anchored at the start of the sample.
  * GSADF (Phillips, Shi & Yu, 2015): sup over all windows [r1, r2] with
    r2 - r1 ≥ r0 (double-sup).
  * BSADF sequence: for each endpoint r2, the backward sup ADF — used for
    real-time date-stamping of bubble origination/termination.
  * Monte-Carlo critical values, including a per-endpoint critical-value
    *sequence* for date-stamping. The null is a driftless unit-root process
    whose changes are iid N(0,1) or follow a given AR model, and the null
    statistics use the same lag order k as the tested statistic.
  * A wild-bootstrap p-value and critical values for GSADF/BSADF. The changes
    are rebuilt recursively from the fitted null autoregression, as in
    Phillips & Shi (2020), with each residual multiplied by a random weight
    at its own date (the time-aligned wild bootstrap of Harvey, Leybourne,
    Sollis & Taylor, 2016). The bootstrap therefore keeps both the short-run
    dynamics and the time profile of volatility, which oversizes the
    Monte-Carlo test. The GSADF critical value is the quantile of the maximum
    over all endpoints; the BSADF sequence is pointwise, per endpoint.

For k = 0 and k = 1, which are the lag orders the simulation loops use, the
windowed ADF statistic is computed in O(1) per window from prefix sums, which
makes the O(T²)-window double-sup feasible inside Monte-Carlo loops. For
k = 0:

    regression sample for window [s, e]:  t = s+1, …, e   (n = e - s obs)
    x_t = y_{t-1},  d_t = Δy_t
    β̂  = (n·S_xd − S_x S_d) / (n·S_xx − S_x²)
    α̂  = (S_d − β̂ S_x) / n
    RSS = S_dd − α̂ S_d − β̂ S_xd          (since RSS = d'd − b̂'X'd)
    se(β̂) = sqrt( σ̂² n / (n·S_xx − S_x²) ),  σ̂² = RSS/(n−2)

For k = 1 the regression sample is t = s+2, …, e (n = e − s − 1) with the
extra regressor z_t = Δy_{t-1}. With intercept-centred moments
C_ab = S_ab − S_a S_b / n, the two-regressor OLS gives

    D  = C_xx C_zz − C_xz²,   β̂ = (C_zz C_xd − C_xz C_zd) / D,
    ψ̂  = (C_xx C_zd − C_xz C_xd) / D,
    RSS = C_dd − β̂ C_xd − ψ̂ C_zd,   se(β̂) = sqrt( σ̂² C_zz / D ),  σ̂² = RSS/(n−3).

All prefix sums are over the full series, so each window statistic is a
handful of float ops. The general k-lag routine solves the normal equations
directly and is used for the observed statistics. Everything is JIT-compiled
with numba; the Monte-Carlo and bootstrap loops are parallelised with prange.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# --------------------------------------------------------------------------
# Windowed ADF statistics
# --------------------------------------------------------------------------


@njit(cache=True)
def _prefix_sums(y):
    """Prefix sums of the cross-moments needed for the k=0 windowed ADF.

    Index convention: sums[t] aggregates observations 1..t of the
    regression sample built on (x_t = y[t-1], d_t = y[t] - y[t-1]).
    """
    T = y.shape[0]
    Sx = np.zeros(T)
    Sxx = np.zeros(T)
    Sd = np.zeros(T)
    Sxd = np.zeros(T)
    Sdd = np.zeros(T)
    for t in range(1, T):
        x = y[t - 1]
        d = y[t] - y[t - 1]
        Sx[t] = Sx[t - 1] + x
        Sxx[t] = Sxx[t - 1] + x * x
        Sd[t] = Sd[t - 1] + d
        Sxd[t] = Sxd[t - 1] + x * d
        Sdd[t] = Sdd[t - 1] + d * d
    return Sx, Sxx, Sd, Sxd, Sdd


@njit(cache=True)
def _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e):
    """ADF t-stat on window [s, e] (0-based, inclusive), k = 0, via prefix sums."""
    n = float(e - s)
    if n < 3.0:
        return -np.inf
    sx = Sx[e] - Sx[s]
    sxx = Sxx[e] - Sxx[s]
    sd = Sd[e] - Sd[s]
    sxd = Sxd[e] - Sxd[s]
    sdd = Sdd[e] - Sdd[s]
    det = n * sxx - sx * sx
    if det <= 1e-12:
        return -np.inf
    beta = (n * sxd - sx * sd) / det
    alpha = (sd - beta * sx) / n
    rss = sdd - alpha * sd - beta * sxd
    if rss <= 0.0:
        return -np.inf
    sigma2 = rss / (n - 2.0)
    se = np.sqrt(sigma2 * n / det)
    if se <= 0.0:
        return -np.inf
    return beta / se


@njit(cache=True)
def _prefix_sums_k1(y):
    """Prefix sums for the k = 1 windowed ADF.

    sums[t] aggregates regression observations 2..t built on
    x_t = y[t-1], z_t = y[t-1] - y[t-2], d_t = y[t] - y[t-1].
    """
    T = y.shape[0]
    out = np.zeros((9, T))  # Sx, Sz, Sd, Sxx, Szz, Sxz, Sxd, Szd, Sdd
    for t in range(2, T):
        x = y[t - 1]
        z = y[t - 1] - y[t - 2]
        d = y[t] - y[t - 1]
        out[0, t] = out[0, t - 1] + x
        out[1, t] = out[1, t - 1] + z
        out[2, t] = out[2, t - 1] + d
        out[3, t] = out[3, t - 1] + x * x
        out[4, t] = out[4, t - 1] + z * z
        out[5, t] = out[5, t - 1] + x * z
        out[6, t] = out[6, t - 1] + x * d
        out[7, t] = out[7, t - 1] + z * d
        out[8, t] = out[8, t - 1] + d * d
    return out


@njit(cache=True)
def _adf_window_k1(P, s, e):
    """ADF t-stat on window [s, e] (0-based, inclusive), k = 1, via prefix sums."""
    n = float(e - s - 1)          # observations t = s+2 .. e
    if n < 5.0:
        return -np.inf
    a, b = e, s + 1
    sx = P[0, a] - P[0, b]
    sz = P[1, a] - P[1, b]
    sd = P[2, a] - P[2, b]
    cxx = (P[3, a] - P[3, b]) - sx * sx / n
    czz = (P[4, a] - P[4, b]) - sz * sz / n
    cxz = (P[5, a] - P[5, b]) - sx * sz / n
    cxd = (P[6, a] - P[6, b]) - sx * sd / n
    czd = (P[7, a] - P[7, b]) - sz * sd / n
    cdd = (P[8, a] - P[8, b]) - sd * sd / n
    det = cxx * czz - cxz * cxz
    if det <= 1e-12 or czz <= 0.0:
        return -np.inf
    beta = (czz * cxd - cxz * czd) / det
    psi = (cxx * czd - cxz * cxd) / det
    rss = cdd - beta * cxd - psi * czd
    if rss <= 0.0:
        return -np.inf
    se = np.sqrt(rss / (n - 3.0) * czz / det)
    if se <= 0.0:
        return -np.inf
    return beta / se


@njit(cache=True)
def _bsadf_fast_k0(y, w0, out):
    """Fill out[e] with BSADF_e for k = 0 (prefix-sum path); return the max."""
    T = y.shape[0]
    Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
    best_all = -np.inf
    for e in range(w0 - 1, T):
        best = -np.inf
        for s in range(0, e - w0 + 2):
            stat = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e)
            if stat > best:
                best = stat
        out[e] = best
        if best > best_all:
            best_all = best
    return best_all


@njit(cache=True)
def _bsadf_fast_k1(y, w0, out):
    """Fill out[e] with BSADF_e for k = 1 (prefix-sum path); return the max."""
    T = y.shape[0]
    P = _prefix_sums_k1(y)
    best_all = -np.inf
    for e in range(w0 - 1, T):
        best = -np.inf
        for s in range(0, e - w0 + 2):
            stat = _adf_window_k1(P, s, e)
            if stat > best:
                best = stat
        out[e] = best
        if best > best_all:
            best_all = best
    return best_all


@njit(cache=True)
def _bsadf_fast(y, w0, k, out):
    """Fill out[e] with BSADF_e (k = 0 or 1, prefix-sum path); return the max."""
    if k == 0:
        return _bsadf_fast_k0(y, w0, out)
    return _bsadf_fast_k1(y, w0, out)


@njit(cache=True)
def _adf_window_klags(y, s, e, k):
    """ADF t-stat on window [s, e] with k lagged differences (intercept, no trend).

    Solves the normal equations directly; p = k + 2 regressors.
    """
    p = k + 2
    n = e - s - k  # usable observations: t = s+k+1 .. e
    if n < p + 2:
        return -np.inf
    XtX = np.zeros((p, p))
    Xty = np.zeros(p)
    yty = 0.0
    for t in range(s + k + 1, e + 1):
        d = y[t] - y[t - 1]
        # regressor vector: [1, y_{t-1}, dy_{t-1}, ..., dy_{t-k}]
        row = np.empty(p)
        row[0] = 1.0
        row[1] = y[t - 1]
        for i in range(1, k + 1):
            row[1 + i] = y[t - i] - y[t - i - 1]
        for a in range(p):
            Xty[a] += row[a] * d
            for b in range(a, p):
                XtX[a, b] += row[a] * row[b]
        yty += d * d
    for a in range(p):
        for b in range(a):
            XtX[a, b] = XtX[b, a]
    # solve
    coef = np.linalg.solve(XtX, Xty)
    rss = yty
    for a in range(p):
        rss -= coef[a] * Xty[a]
    if rss <= 0.0:
        return -np.inf
    sigma2 = rss / (n - p)
    XtX_inv = np.linalg.inv(XtX)
    var_b = sigma2 * XtX_inv[1, 1]
    if var_b <= 0.0:
        return -np.inf
    return coef[1] / np.sqrt(var_b)


def adf_stat(y: np.ndarray, k: int = 0, s: int = 0, e: int | None = None) -> float:
    """Public single-window ADF statistic (intercept, k lagged differences)."""
    y = np.asarray(y, dtype=np.float64)
    if e is None:
        e = len(y) - 1
    if k == 0:
        sums = _prefix_sums(y)
        return _adf_window_k0(*sums, s, e)
    return _adf_window_klags(y, s, e, k)


# --------------------------------------------------------------------------
# BSADF / GSADF / SADF
# --------------------------------------------------------------------------


@njit(cache=True)
def _bsadf_sequence_k0(y, w0):
    """BSADF_t for every endpoint t ≥ w0-1, k = 0 (prefix-sum fast path)."""
    T = y.shape[0]
    Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        best = -np.inf
        for s in range(0, e - w0 + 2):
            stat = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, s, e)
            if stat > best:
                best = stat
        out[e] = best
    return out


@njit(cache=True)
def _bsadf_sequence_klags(y, w0, k):
    T = y.shape[0]
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        best = -np.inf
        for s in range(0, e - w0 + 2):
            stat = _adf_window_klags(y, s, e, k)
            if stat > best:
                best = stat
        out[e] = best
    return out


@njit(cache=True)
def _sadf_sequence_k0(y, w0):
    """Forward-expanding ADF (anchored at 0) for every endpoint — PWY (2011)."""
    T = y.shape[0]
    Sx, Sxx, Sd, Sxd, Sdd = _prefix_sums(y)
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        out[e] = _adf_window_k0(Sx, Sxx, Sd, Sxd, Sdd, 0, e)
    return out


@njit(cache=True)
def _sadf_sequence_klags(y, w0, k):
    """Forward-expanding ADF with k lagged differences for every endpoint."""
    T = y.shape[0]
    out = np.full(T, -np.inf)
    for e in range(w0 - 1, T):
        out[e] = _adf_window_klags(y, 0, e, k)
    return out


def min_window(T: int) -> int:
    """PSY (2015) rule of thumb: r0 = 0.01 + 1.8/√T, w0 = ⌈r0·T⌉."""
    return int(np.ceil(T * (0.01 + 1.8 / np.sqrt(T))))


def bsadf(y: np.ndarray, k: int = 0, w0: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if w0 is None:
        w0 = min_window(len(y))
    if k == 0:
        return _bsadf_sequence_k0(y, w0)
    return _bsadf_sequence_klags(y, w0, k)


def gsadf(y: np.ndarray, k: int = 0, w0: int | None = None) -> float:
    return float(np.max(bsadf(y, k=k, w0=w0)))


def sadf(y: np.ndarray, k: int = 0, w0: int | None = None) -> float:
    y = np.asarray(y, dtype=np.float64)
    if w0 is None:
        w0 = min_window(len(y))
    if k == 0:
        return float(np.max(_sadf_sequence_k0(y, w0)))
    return float(np.max(_sadf_sequence_klags(y, w0, k)))


# --------------------------------------------------------------------------
# Monte-Carlo critical values (null: driftless random walk)
# --------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _mc_null_bsadf(T, w0, n_reps, seed, k, psi):
    """BSADF_t sequences under H0: y_t = y_{t-1} + u_t with
    u_t = Σ_i psi_i u_{t-i} + ε_t, ε ~ iid N(0,1) (u = ε when psi is empty).

    The null statistics use k lagged differences (k = 0 or 1).
    Returns the (n_reps, T) matrix of null BSADF sequences.
    """
    p = psi.shape[0]
    out = np.full((n_reps, T), -np.inf)
    for r in prange(n_reps):
        np.random.seed(seed + r)
        eps = np.random.standard_normal(T)
        u = np.empty(T)
        for t in range(T):
            v = eps[t]
            for i in range(1, p + 1):
                if t - i >= 0:
                    v += psi[i - 1] * u[t - i]
            u[t] = v
        y = np.cumsum(u)
        _bsadf_fast(y, w0, k, out[r])
    return out


def mc_critical_values(T: int, w0: int | None = None, n_reps: int = 999,
                       quantiles=(0.90, 0.95, 0.99), seed: int = 42,
                       k: int = 0, psi=None):
    """Monte-Carlo critical values for GSADF and the BSADF date-stamping sequence.

    k is the lag order of the null ADF regressions (0 or 1) and should match
    the tested statistic. psi, if given, are AR coefficients for the changes
    of the unit-root null; the default is an iid random walk.

    Returns
    -------
    dict with:
      'gsadf_cv'    : {q: scalar} critical values for the global GSADF stat
      'bsadf_cv_seq': {q: (T,) array} per-endpoint critical values for BSADF_t
      'null_gsadf'  : (n_reps,) simulated null GSADF distribution
    """
    if k not in (0, 1):
        raise ValueError("Monte-Carlo nulls support k = 0 or 1")
    if w0 is None:
        w0 = min_window(T)
    psi = np.zeros(0) if psi is None else np.asarray(psi, dtype=np.float64)
    null_seq = _mc_null_bsadf(T, w0, n_reps, seed, k, psi)
    null_gsadf = np.max(null_seq, axis=1)
    gsadf_cv = {q: float(np.quantile(null_gsadf, q)) for q in quantiles}
    bsadf_cv_seq = {}
    for q in quantiles:
        cv = np.full(T, np.nan)
        cv[w0 - 1:] = np.quantile(null_seq[:, w0 - 1:], q, axis=0)
        bsadf_cv_seq[q] = cv
    return {"gsadf_cv": gsadf_cv, "bsadf_cv_seq": bsadf_cv_seq,
            "null_gsadf": null_gsadf, "w0": w0, "k": k, "n_reps": n_reps,
            "psi": psi.tolist()}


# --------------------------------------------------------------------------
# Wild bootstrap (AR null as in Phillips & Shi, 2020; time-aligned weights)
# --------------------------------------------------------------------------


def fit_null_ar(y: np.ndarray, p: int):
    """OLS of Δy_t on an intercept and p own lags (the unit-root null).

    Returns (psi, resid): psi has p coefficients; resid (length T-1-p) are the
    residuals for t = p+1 … T-1 (in Δy indexing), mean zero by construction.
    """
    dy = np.diff(np.asarray(y, dtype=np.float64))
    if p == 0:
        return np.zeros(0), dy - dy.mean()
    X = np.column_stack([np.ones(len(dy) - p)]
                        + [dy[p - i:len(dy) - i] for i in range(1, p + 1)])
    coef = np.linalg.lstsq(X, dy[p:], rcond=None)[0]
    return coef[1:], dy[p:] - X @ coef


@njit(cache=True, parallel=True)
def _wild_bootstrap_gsadf(dy_init, psi, resid, w0, n_reps, seed, k):
    """Wild bootstrap of the GSADF statistic and BSADF sequence.

    Bootstrap changes follow Δy*_t = Σ_i psi_i Δy*_{t-i} + w_t ê_t, with
    w_t iid N(0,1), the first p changes set to their observed values
    (dy_init), and no drift; y*_t accumulates them. With p = 0 this is
    y*_t = Σ_{s≤t} w_s ê_s on the demeaned changes. Each residual keeps its
    own date (Harvey, Leybourne, Sollis & Taylor, 2016), so the resampled
    series keeps the time profile of the observed volatility; the fitted
    autoregression (Phillips & Shi, 2020) keeps its short-run dynamics.
    Statistics use k lagged differences (k = 0 or 1).
    Returns (n_reps,) null GSADF stats and (n_reps, T) BSADF sequences.
    """
    p = psi.shape[0]
    n_e = resid.shape[0]
    T = n_e + p + 1
    stats = np.empty(n_reps)
    seqs = np.full((n_reps, T), -np.inf)
    for r in prange(n_reps):
        np.random.seed(seed + 7919 * r)
        w = np.random.standard_normal(n_e)
        dy = np.empty(T - 1)
        for t in range(p):
            dy[t] = dy_init[t]
        for j in range(n_e):
            t = p + j
            v = w[j] * resid[j]
            for i in range(1, p + 1):
                v += psi[i - 1] * dy[t - i]
            dy[t] = v
        y = np.empty(T)
        y[0] = 0.0
        for t in range(1, T):
            y[t] = y[t - 1] + dy[t - 1]
        stats[r] = _bsadf_fast(y, w0, k, seqs[r])
    return stats, seqs


def wild_bootstrap_gsadf(y: np.ndarray, k: int = 0, w0: int | None = None,
                         n_reps: int = 499, seed: int = 7,
                         quantiles=(0.90, 0.95, 0.99), ar_lags: int | None = None):
    """Wild-bootstrap p-value for GSADF + bootstrap BSADF critical-value sequence.

    k is the lag order of the ADF regressions (observed and bootstrap, 0 or 1).
    ar_lags is the order of the null autoregression for Δy used to rebuild the
    bootstrap changes (default k, as in Phillips & Shi, 2020). The BSADF
    critical-value sequence is pointwise (per endpoint), with no correction
    for testing many endpoints; gsadf_cv, the quantile of each draw's maximum,
    is the whole-sample (family-wise) critical value.
    """
    if k not in (0, 1):
        raise ValueError("the bootstrap supports k = 0 or 1")
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if w0 is None:
        w0 = min_window(T)
    p = k if ar_lags is None else ar_lags
    psi, resid = fit_null_ar(y, p)
    dy_init = np.diff(y)[:p].copy()
    null_stats, null_seqs = _wild_bootstrap_gsadf(dy_init, psi, resid, w0,
                                                  n_reps, seed, k)
    obs = gsadf(y, k=k, w0=w0)
    pval = float((np.sum(null_stats >= obs) + 1) / (n_reps + 1))
    bsadf_cv_seq = {}
    for q in quantiles:
        cv = np.full(T, np.nan)
        cv[w0 - 1:] = np.quantile(null_seqs[:, w0 - 1:], q, axis=0)
        bsadf_cv_seq[q] = cv
    return {"gsadf_obs": obs, "pvalue": pval, "null_gsadf": null_stats,
            "gsadf_cv": {q: float(np.quantile(null_stats, q)) for q in quantiles},
            "bsadf_cv_seq": bsadf_cv_seq, "w0": w0, "k": k, "ar_lags": p,
            "psi": psi.tolist(), "n_reps": n_reps}


# --------------------------------------------------------------------------
# Date-stamping
# --------------------------------------------------------------------------


def date_stamp(bsadf_seq: np.ndarray, cv_seq: np.ndarray, index,
               min_duration: int | None = None):
    """PSY date-stamping: episodes where BSADF_t > cv_t for ≥ min_duration periods.

    min_duration defaults to ⌈log(T)⌉ observations (PSY's δ·log(T) rule with
    δ = 1), filtering out one-month blips.
    Returns list of dicts {start, end, peak, peak_stat, n_periods}.
    """
    T = len(bsadf_seq)
    if min_duration is None:
        min_duration = int(np.ceil(np.log(T)))
    above = (bsadf_seq > cv_seq) & np.isfinite(bsadf_seq) & np.isfinite(cv_seq)
    episodes = []
    t = 0
    while t < T:
        if above[t]:
            start = t
            while t < T and above[t]:
                t += 1
            end = t - 1  # inclusive
            if end - start + 1 >= min_duration:
                seg = bsadf_seq[start:end + 1]
                peak_rel = int(np.argmax(seg))
                episodes.append({
                    "start": index[start], "end": index[end],
                    "peak": index[start + peak_rel],
                    "peak_stat": float(seg[peak_rel]),
                    "n_periods": int(end - start + 1),
                })
        else:
            t += 1
    return episodes
