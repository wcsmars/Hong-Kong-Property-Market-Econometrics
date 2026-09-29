"""
longrun.py — Long-run equilibrium analysis: unit roots, cointegration,
error-correction, and structural breaks.

Methods:
  * Unit-root pre-tests: ADF (intercept / intercept+trend), KPSS as
    confirmatory, Zivot-Andrews allowing one endogenous break.
  * Engle-Granger two-step cointegration with MacKinnon p-values.
  * Johansen trace / max-eigenvalue tests (statsmodels VECM machinery).
  * ARDL / UECM bounds test (Pesaran, Shin & Smith 2001) — valid whether
    regressors are I(0) or I(1); lag order by AIC with every regressor kept;
    critical values and p-values for k = number of regressors.
  * Dynamic OLS (Stock-Watson) long-run coefficients with leads/lags and
    Newey-West standard errors — superconsistent and median-unbiased.
  * Error-correction model with half-life of deviations.
  * Bai-Perron-style multiple structural breaks on the equilibrium error
    via dynamic programming (ruptures, l2 cost, BIC-penalised).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.tsa.stattools import adfuller, kpss, coint, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order, pss_critical_values
from statsmodels.tsa.ardl.model import _pss_pvalue
import statsmodels.api as sm

# ---------------------------------------------------------------- unit roots


def unit_root_table(df: pd.DataFrame, vars_: list[str]) -> pd.DataFrame:
    """ADF (level & first difference), KPSS, Zivot-Andrews for each variable."""
    rows = []
    for v in vars_:
        x = df[v].dropna()
        adf_l = adfuller(x, regression="c", autolag="AIC")
        adf_t = adfuller(x, regression="ct", autolag="AIC")
        adf_d = adfuller(x.diff().dropna(), regression="c", autolag="AIC")
        try:
            kp = kpss(x, regression="c", nlags="auto")
        except Exception:
            kp = (np.nan, np.nan)
        try:
            za = zivot_andrews(x, regression="c", autolag="AIC")
            za_stat, za_p = za[0], za[1]
            za_break = x.index[za[4]] if isinstance(za[4], (int, np.integer)) else za[4]
        except Exception:
            za_stat, za_p, za_break = np.nan, np.nan, None
        rows.append({
            "variable": v,
            "ADF_level": adf_l[0], "ADF_level_p": adf_l[1],
            "ADF_trend": adf_t[0], "ADF_trend_p": adf_t[1],
            "ADF_diff": adf_d[0], "ADF_diff_p": adf_d[1],
            "KPSS_level": kp[0], "KPSS_p": kp[1],
            "ZA_stat": za_stat, "ZA_p": za_p, "ZA_break": za_break,
        })
    return pd.DataFrame(rows).set_index("variable")


# ------------------------------------------------------------- cointegration


def engle_granger(y: pd.Series, X: pd.DataFrame):
    """EG test of no-cointegration + first-stage static OLS coefficients."""
    data = pd.concat([y, X], axis=1).dropna()
    stat, pval, crit = coint(data.iloc[:, 0], data.iloc[:, 1:],
                             trend="c", autolag="aic")
    ols = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])).fit()
    resid = ols.resid
    return {"eg_stat": stat, "eg_p": pval, "eg_crit": dict(zip(["1%", "5%", "10%"], crit)),
            "static_beta": ols.params, "resid": resid}


def johansen(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 2):
    """Johansen trace & max-eig tests with k_ar_diff lagged differences.

    det_order=0 is statsmodels' unrestricted constant (a constant in the VECM,
    allowing linear trends in the levels); the critical values match that case.
    """
    res = coint_johansen(data.dropna(), det_order, k_ar_diff)
    out = []
    for r in range(data.shape[1]):
        out.append({
            "r": r,
            "trace": res.lr1[r], "trace_cv95": res.cvt[r, 1],
            "maxeig": res.lr2[r], "maxeig_cv95": res.cvm[r, 1],
            "trace_reject": res.lr1[r] > res.cvt[r, 1],
            "maxeig_reject": res.lr2[r] > res.cvm[r, 1],
        })
    # normalised first cointegrating vector (on the first variable)
    beta = res.evec[:, 0] / res.evec[0, 0]
    return pd.DataFrame(out).set_index("r"), pd.Series(beta, index=data.columns)


# PSS (2001) Table CII(iii), case III (unrestricted intercept, no trend):
# 5% bounds for the t-statistic on the lagged level of y, by regressor count
PSS_T_BOUNDS_5PCT = {1: (-2.86, -3.22), 2: (-2.86, -3.53), 3: (-2.86, -3.78),
                     4: (-2.86, -3.99)}


def pss_bounds(stat: float, k: int, case: int = 3):
    """PSS (2001) asymptotic bounds critical values and p-values for k regressors.

    statsmodels' UECMResults.bounds_test sets k = len(ardl_order), which also
    counts the dependent variable, so its bounds are one regressor too
    lenient. Here k is the number of long-run regressors, excluding y.
    Returns (crit, pvals): crit has rows 90/95/99/99.9 and columns
    lower (all regressors I(0)) / upper (all I(1)); pvals has the same columns.
    """
    cv = pss_critical_values.crit_vals
    crit = pd.DataFrame({"lower": cv[(k, case, False)], "upper": cv[(k, case, True)]},
                        index=pss_critical_values.crit_percentiles)
    crit.index.name = "percentile"
    pvals = pd.Series({"lower": _pss_pvalue(stat, k, case, False),
                       "upper": _pss_pvalue(stat, k, case, True)})
    return crit, pvals


def ardl_bounds(y: pd.Series, X: pd.DataFrame, maxlag: int = 6, trend: str = "c"):
    """PSS (2001) bounds test via UECM with AIC lag selection.

    Lag orders are chosen by AIC among models that keep every regressor
    (order >= 0): the bounds test concerns the specified long-run relation,
    so AIC may not drop a regressor from it. Critical values and p-values
    use k = number of regressors (see pss_bounds).
    """
    data = pd.concat([y, X], axis=1).dropna()
    endog = data.iloc[:, 0]
    exog = data.iloc[:, 1:]
    sel = ardl_select_order(endog, maxlag, exog, maxlag, trend=trend, ic="aic")
    # sel.aic lists every candidate by ascending AIC as (AR lags, {regressor:
    # order}); an order of None means that regressor was left out entirely.
    ar, dl = next(orders for orders in sel.aic
                  if all(q is not None for q in orders[1].values()))
    ardl = ARDL(endog, ar, exog, {c: dl[c] for c in exog.columns}, trend=trend)
    ardl_res = ardl.fit()
    # UECM requires every exog to enter with lag >= 1; clamp the
    # AIC-selected orders accordingly before building the UECM directly.
    p = max(int(ar or 0), 1)
    qs = {c: max(int(dl[c]), 1) for c in exog.columns}
    uecm = UECM(endog, lags=p, exog=exog, order=qs, trend=trend)
    uecm_res = uecm.fit()
    case = 3  # unrestricted intercept, no trend
    stat = float(uecm_res.bounds_test(case=case).stat)
    crit, pvals = pss_bounds(stat, k=exog.shape[1], case=case)
    # PSS t-bounds statistic: OLS t-ratio on the lagged level of y in the UECM
    t_stat = float(uecm_res.tvalues[f"{endog.name}.L1"])
    return {"ardl_order": ardl.ardl_order, "k": exog.shape[1], "bounds_stat": stat,
            "bounds_t": t_stat, "bounds_crit": crit, "bounds_pvals": pvals,
            "uecm_res": uecm_res, "ardl_res": ardl_res}


def dols(y: pd.Series, X: pd.DataFrame, leads_lags: int = 4, hac_lags: int = 6):
    """Stock-Watson Dynamic OLS: y_t = c + β'x_t + Σ_{j=-p..p} γ_j'Δx_{t+j} + u_t."""
    data = pd.concat([y, X], axis=1).dropna()
    yv = data.iloc[:, 0]
    Xl = data.iloc[:, 1:]
    parts = [Xl]
    for j in range(-leads_lags, leads_lags + 1):
        dx = Xl.diff().shift(-j)
        dx.columns = [f"d_{c}_ll{j}" for c in Xl.columns]
        parts.append(dx)
    Z = pd.concat(parts, axis=1)
    aligned = pd.concat([yv, Z], axis=1).dropna()
    res = sm.OLS(aligned.iloc[:, 0], sm.add_constant(aligned.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    beta = res.params[Xl.columns]
    se = res.bse[Xl.columns]
    resid_lr = yv - (res.params["const"] + (Xl @ beta))
    return {"beta": beta, "se": se, "res": res, "lr_resid": resid_lr.dropna()}


def ecm(y: pd.Series, X: pd.DataFrame, lr_resid: pd.Series, n_lags: int = 2,
        hac_lags: int = 6):
    """Single-equation ECM: Δy_t on ect_{t-1}, lagged Δy, contemporaneous+lagged ΔX.

    Returns fitted results + speed of adjustment λ and half-life ln(0.5)/ln(1+λ).
    """
    dy = y.diff()
    dX = X.diff()
    ect = lr_resid.shift(1).rename("ect_l1")
    parts = {"ect_l1": ect}
    for l in range(1, n_lags + 1):
        parts[f"dy_l{l}"] = dy.shift(l)
    for c in dX.columns:
        parts[f"d_{c}"] = dX[c]
        for l in range(1, n_lags + 1):
            parts[f"d_{c}_l{l}"] = dX[c].shift(l)
    Z = pd.DataFrame(parts)
    aligned = pd.concat([dy.rename("dy"), Z], axis=1).dropna()
    res = sm.OLS(aligned["dy"], sm.add_constant(aligned.drop(columns="dy"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    lam = res.params["ect_l1"]
    # no finite half-life unless -1 < λ < 0 (None keeps the JSON output strict)
    half_life = float(np.log(0.5) / np.log(1.0 + lam)) if -1 < lam < 0 else None
    return {"res": res, "lambda": float(lam), "lambda_t": float(res.tvalues["ect_l1"]),
            "half_life_months": half_life}


# ------------------------------------------------------------------- breaks


def bai_perron_breaks(x: pd.Series, max_breaks: int = 5, min_size: int = 24):
    """Multiple mean-shift breaks by dynamic programming, BIC-selected count.

    Operates on the supplied series (typically the equilibrium error or
    log-returns). Pure change-in-mean specification, l2 cost — the
    Bai-Perron (1998) pure structural change model estimated by exact DP.
    BIC counts m + 1 segment means and m break dates. When the selected count
    equals max_breaks ("capped"), BIC would add more breaks, as it does for a
    highly persistent series, so the dates describe that persistence rather
    than a few discrete regime shifts.
    """
    z = x.dropna()
    v = z.values.astype(float)
    n = len(v)
    algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(v.reshape(-1, 1))
    best = {"n_breaks": 0, "bic": np.inf, "bkps": []}
    sigma2_full = np.var(v)
    for m in range(0, max_breaks + 1):
        if m == 0:
            rss = float(np.sum((v - v.mean()) ** 2))
            bkps = []
        else:
            try:
                ends = algo.predict(n_bkps=m)
            except Exception:
                continue
            rss = 0.0
            start = 0
            for e in ends:
                seg = v[start:e]
                rss += float(np.sum((seg - seg.mean()) ** 2))
                start = e
            bkps = ends[:-1]
        n_params = (m + 1) + m  # segment means and break dates
        bic = n * np.log(rss / n + 1e-300) + n_params * np.log(n)
        if bic < best["bic"]:
            best = {"n_breaks": m, "bic": bic, "bkps": bkps}
    best["break_dates"] = [z.index[b] for b in best["bkps"]]
    best["capped"] = best["n_breaks"] == max_breaks
    best["sigma2_full"] = sigma2_full
    return best


# ------------------------------------------------- Gregory-Hansen (1996)

# Critical values from Gregory & Hansen (1996, J. Econometrics 70, Table 1),
# ADF* statistic; m = number of RHS regressors (excluding deterministics).
# Verified cell-by-cell against the published table (p. 109 of the original
# article) and the tspdlib reference implementation on 2026-06-13.
GH_CV = {
    "C":  {1: {-0.01: -5.13, 0.05: -4.61, 0.10: -4.34},
           2: {-0.01: -5.44, 0.05: -4.92, 0.10: -4.69},
           3: {-0.01: -5.77, 0.05: -5.28, 0.10: -5.02},
           4: {-0.01: -6.05, 0.05: -5.56, 0.10: -5.31}},
    "CT": {1: {-0.01: -5.45, 0.05: -4.99, 0.10: -4.72},
           2: {-0.01: -5.80, 0.05: -5.29, 0.10: -5.03},
           3: {-0.01: -6.05, 0.05: -5.57, 0.10: -5.33},
           4: {-0.01: -6.36, 0.05: -5.83, 0.10: -5.59}},
    "CS": {1: {-0.01: -5.47, 0.05: -4.95, 0.10: -4.68},
           2: {-0.01: -5.97, 0.05: -5.50, 0.10: -5.23},
           3: {-0.01: -6.51, 0.05: -6.00, 0.10: -5.75},
           4: {-0.01: -6.92, 0.05: -6.41, 0.10: -6.17}},
}


def gregory_hansen(y: pd.Series, X: pd.DataFrame, model: str = "CS",
                   trim: float = 0.15):
    """Gregory-Hansen test for cointegration with one endogenous break.

    model: 'C' level shift, 'CT' level shift + trend, 'CS' regime shift
    (break in intercept AND slopes). Statistic: inf over break dates of the
    ADF t-stat (no deterministics, AIC lags) on the regression residuals.
    Returns inf-ADF stat, break date minimising it, cv table, and the
    residuals/coefficients at the chosen break.
    """
    data = pd.concat([y, X], axis=1).dropna()
    yv = data.iloc[:, 0].values
    Xv = data.iloc[:, 1:].values
    n, m = Xv.shape
    if m > 4:
        raise ValueError("Gregory-Hansen critical values are tabulated only "
                         "for m <= 4 regressors")
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n))
    best = {"stat": np.inf, "tau": None}
    trend = np.arange(n, dtype=float)
    for tau in range(lo, hi):
        D = (np.arange(n) >= tau).astype(float)
        cols = [np.ones(n), D]
        if model == "CT":
            cols.append(trend)
        cols.append(Xv)
        if model == "CS":
            cols.append(Xv * D[:, None])
        Z = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
        resid = yv - Z @ beta
        stat = adfuller(resid, regression="n", autolag="AIC", maxlag=6)[0]
        if stat < best["stat"]:
            best = {"stat": float(stat), "tau": tau, "beta": beta,
                    "resid": pd.Series(resid, index=data.index)}
    best["break_date"] = data.index[best["tau"]]
    best["cv"] = GH_CV[model][m]
    best["model"] = model
    best["reject_5pct"] = best["stat"] < best["cv"][0.05]
    return best
