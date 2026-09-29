"""
longrun.py — The cap-rate channel.

(a) Unit-root battery (ADF, DF-GLS, Phillips-Perron, KPSS, Zivot-Andrews)
    on office rents, prices, price-rent ratios and real rates.
(b) Cointegration: Johansen trace, ARDL/UECM bounds (PSS 2001), and
    hand-rolled DOLS (Stock-Watson 1993) with HAC standard errors, for the
    Grade A (and overall) log price-rent ratio against real interest rates.
(c) Campbell-Shiller log-linear present-value decomposition of the office
    price-rent ratio on quarterly data (1986Q1+), with the discount-rate state
    proxied by the real Best Lending Rate (available the full sample), and a
    residual-based VAR bootstrap for the variance-decomposition shares.

Outputs: output/unit_roots.csv, output/johansen.csv, output/longrun.json,
         output/longrun_log.txt
Missing-data policy: the 5-8 unpublished monthly price-index observations
('^', <5 transactions) are log-linearly interpolated for monthly estimations
and counted in the output; quarterly series are complete and serve as the
robustness frequency.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron, ZivotAndrews
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ardl import UECM, pss_critical_values
from statsmodels.tsa.ardl.model import _pss_pvalue
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
LOG: list[str] = []


def log(msg: str):
    LOG.append(str(msg))
    print(msg)


def load_monthly() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(PRO, "monthly_panel.csv"),
                    index_col=0, parse_dates=True)
    n_interp = {}
    for c in ("lp_A", "lp_ALL", "lpr_A", "lpr_ALL", "lp_real_A", "lp_real_ALL"):
        n = p[c].isna().sum()
        first, last = p[c].first_valid_index(), p[c].last_valid_index()
        p[c] = p[c].interpolate("linear", limit_area="inside")
        n_interp[c] = int(n - p[c].isna().sum())
        assert p.loc[first:last, c].notna().all()
    p.attrs["n_interp"] = n_interp
    return p


def load_quarterly() -> pd.DataFrame:
    q = pd.read_csv(os.path.join(PRO, "quarterly_indices.csv"), index_col=0)
    q.index = pd.PeriodIndex(q.index, freq="Q")
    return q


# ------------------------------------------------------------- unit roots
def unit_root_battery(panel: pd.DataFrame) -> pd.DataFrame:
    series = {
        "lr_real_A": panel["lr_real_A"], "lp_real_A": panel["lp_real_A"],
        "lpr_A": panel["lpr_A"], "lpr_ALL": panel["lpr_ALL"],
        "real_hibor": panel["real_hibor"], "real_gs10": panel["real_gs10"],
        "log_yld_A": np.log(panel["yld_A_mean"]),
    }
    rows = []
    for name, s in series.items():
        for diff in (0, 1):
            x = (s.diff().dropna() if diff else s.dropna()).values
            lbl = f"d_{name}" if diff else name
            try:
                adf = ADF(x, trend="c")
                dfg = DFGLS(x, trend="c")
                pp = PhillipsPerron(x, trend="c")
                kp = KPSS(x, trend="c")
                rows.append({
                    "series": lbl, "n": len(x),
                    "ADF": adf.stat, "ADF_p": adf.pvalue,
                    "DFGLS": dfg.stat, "DFGLS_p": dfg.pvalue,
                    "PP": pp.stat, "PP_p": pp.pvalue,
                    "KPSS": kp.stat, "KPSS_p": kp.pvalue,
                })
            except Exception as e:  # pragma: no cover
                rows.append({"series": lbl, "n": len(x), "err": str(e)})
        # Zivot-Andrews on levels only (allows one endogenous break)
        try:
            za = ZivotAndrews(s.dropna().values, trend="c")
            rows[-2]["ZA"] = za.stat
            rows[-2]["ZA_p"] = za.pvalue
        except Exception:
            pass
    return pd.DataFrame(rows)


# ----------------------------------------------------------- cointegration
def johansen_table(panel: pd.DataFrame) -> pd.DataFrame:
    systems = {
        "S1_lprA_rhibor": ["lpr_A", "real_hibor"],
        "S2_lprALL_rhibor": ["lpr_ALL", "real_hibor"],
        "S3_lpA_lrA_rhibor": ["lp_real_A", "lr_real_A", "real_hibor"],
        "S4_lprA_rgs10": ["lpr_A", "real_gs10"],
    }
    rows = []
    for name, cols in systems.items():
        df = panel[cols].dropna()
        # lag order from levels VAR (AIC), capped at 12
        sel = VAR(df.values).select_order(12)
        k = max(int(sel.aic), 2)
        res = coint_johansen(df.values, det_order=0, k_ar_diff=k - 1)
        for r in range(len(cols)):
            rows.append({
                "system": name, "n": len(df), "k_ar": k, "H0_rank<=": r,
                "trace": res.lr1[r], "cv_95": res.cvt[r, 1],
                "reject": bool(res.lr1[r] > res.cvt[r, 1]),
            })
        if name == "S1_lprA_rhibor":
            # first cointegrating vector normalized as lpr + b·rate ~ I(0), so
            # the implied long-run semi-elasticity of lpr on the rate is −b
            # (same sign convention as the ARDL and DOLS estimates); only
            # meaningful where rank 0 is rejected
            beta = res.evec[:, 0]
            beta = beta / beta[0]
            rows[-2]["lr_semi_elasticity"] = float(-beta[1])
    return pd.DataFrame(rows)


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


def uecm_bounds(panel: pd.DataFrame, y: str, x: str) -> dict:
    df = panel[[y, x]].dropna()
    mod = UECM(df[y], lags=2, exog=df[[x]], order=2)
    res = mod.fit()
    stat = float(res.bounds_test(case=3).stat)
    crit, pvals = pss_bounds(stat, k=1, case=3)  # one regressor, x
    ec = float(res.params[f"{y}.L1"])  # coefficient on y_{t-1} (EC term)
    # long-run semi-elasticity: -(coef on x level)/(coef on y level)
    lr = float(-res.params[f"{x}.L1"] / ec)
    hl = float(np.log(0.5) / np.log(1 + ec)) if -1 < ec < 0 else np.nan
    return {
        "y": y, "x": x, "n": int(len(df)),
        "F_PSS": stat, "k": 1,
        "F_p_lower": float(pvals["lower"]),
        "F_p_upper": float(pvals["upper"]),
        "F_cv95_lower": float(crit.loc[95.0, "lower"]),
        "F_cv95_upper": float(crit.loc[95.0, "upper"]),
        "ec_coef": ec, "ec_t": float(res.tvalues.iloc[1]),
        "longrun_semi_elasticity": lr, "half_life_months": hl,
    }


def dols(panel: pd.DataFrame, y: str, x: str, leads: int = 4,
         hac_lags: int = 12) -> dict:
    """Stock-Watson DOLS: y_t = a + b x_t + sum_{j=-p..p} c_j Δx_{t+j} + u."""
    df = panel[[y, x]].dropna().copy()
    dx = df[x].diff()
    X = pd.DataFrame({x: df[x]})
    for j in range(-leads, leads + 1):
        X[f"dx_{j}"] = dx.shift(-j)
    data = pd.concat([df[y].rename("y"), X], axis=1).dropna()
    mod = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y")))
    res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {
        "y": y, "x": x, "n": int(res.nobs),
        "beta": float(res.params[x]), "se_HAC": float(res.bse[x]),
        "t": float(res.tvalues[x]),
    }


# ------------------------------------------- Campbell-Shiller decomposition
def campbell_shiller(q: pd.DataFrame, panel: pd.DataFrame,
                     n_boot: int = 1000, seed: int = 42) -> dict:
    """Quarterly log-linear present-value decomposition for Grade A offices.

    pd_t = log(P/R) (price-rent); identity:
      pd_t ≈ k/(1-ρ) + E_t Σ ρ^j [Δd_{t+1+j} − r_{t+1+j}]
    where Δd = real rent growth, r = real discount rate (real BLR / 400 per
    quarter). ρ anchored on the mean Grade A office yield (RVD levels).
    Shares: cov(pd, pd_cf)/var(pd) and −cov(pd, pd_dr)/var(pd) from a VAR.
    """
    # quarterly CPI and BLR
    blr_m = panel["blr"]
    # build from raw monthly panel index (1993+) is too short — reload rates
    rates = pd.read_csv(os.path.join(ROOT, "data", "raw", "hkma",
                                     "hkd_retail_rates_monthly_period_average.csv"))
    blr = pd.Series(rates["best_lending_rate"].values,
                    index=pd.PeriodIndex(rates["end_of_month"], freq="M"))
    blr_q = blr.groupby(blr.index.asfreq("Q")).mean()
    cpi_q = q["cpi"]
    infl_q = 100.0 * (cpi_q / cpi_q.shift(4) - 1.0)
    real_blr_q = (blr_q - infl_q).dropna() / 400.0   # per-quarter real rate

    d = np.log(q["rt_A"]).rename("ld")               # log rent index
    p = np.log(q["px_A"]).rename("lp")
    dd = (d - d.shift(1) - np.log(cpi_q / cpi_q.shift(1))).rename("dd")  # real
    pdr = (p - d).rename("pd")                       # log price-rent (index)

    # anchor: mean quarterly yield from RVD average rent/price levels (1999+)
    yld_m = panel["yld_A_mean"].dropna()             # annualized
    ybar_q = float(yld_m.mean()) / 4.0
    rho = 1.0 / (1.0 + ybar_q)

    z = pd.concat([dd, real_blr_q.rename("rf"), pdr], axis=1).dropna()
    zm = z - z.mean()
    sel = VAR(zm.values).select_order(8)
    k = max(int(sel.aic), 1)
    var = VAR(zm.values).fit(k)

    def decompose(varres, zdf) -> tuple[float, float, float]:
        """Cochrane-style shares of var(pd):
        share_rent  = cov(pd, E_t sum rho^j dd_{t+1+j}) / var(pd)
        share_ret   = 1 - share_rent  (returns implied by the PV identity)
        share_rf    = -cov(pd, E_t sum rho^j rf_{t+1+j}) / var(pd)
        premium     = share_ret - share_rf.
        State z~_t = (z_t, ..., z_{t-p+1}); expectations via companion A:
        E_t sum_{j>=0} rho^j z_{t+1+j} = A (I - rho A)^{-1} z~_t.
        """
        K = zdf.shape[1]
        p_ = varres.k_ar
        A = np.zeros((K * p_, K * p_))
        A[:K, :] = np.hstack(varres.coefs)
        if p_ > 1:
            A[K:, :-K] = np.eye(K * (p_ - 1))
        e_d = np.zeros(K * p_); e_d[0] = 1.0
        e_rf = np.zeros(K * p_); e_rf[1] = 1.0
        B = A @ np.linalg.inv(np.eye(K * p_) - rho * A)
        # state aligned at t: rows t = p_-1 .. T-1 (0-based), z~_t stacks z_{t-i}
        Zt = np.column_stack([zdf.values[p_ - 1 - i: len(zdf) - i]
                              if i else zdf.values[p_ - 1:]
                              for i in range(p_)])
        pd_obs = zdf.values[p_ - 1:, 2]
        cf = e_d @ B @ Zt.T
        rf = e_rf @ B @ Zt.T
        v = np.var(pd_obs)
        s_rent = float(np.cov(pd_obs, cf, ddof=0)[0, 1] / v)
        s_rf = float(-np.cov(pd_obs, rf, ddof=0)[0, 1] / v)
        return s_rent, 1.0 - s_rent, s_rf

    s_rent, s_ret, s_rf = decompose(var, zm)

    rng = np.random.default_rng(seed)
    boots = []
    resid = var.resid
    fitted_lags = zm.values[: var.k_ar]
    for _ in range(n_boot):
        idx = rng.integers(0, len(resid), len(resid))
        e = resid[idx]
        sim = list(fitted_lags)
        for t in range(len(e)):
            nxt = (var.intercept
                   + sum(var.coefs[i] @ sim[-1 - i] for i in range(var.k_ar)) + e[t])
            sim.append(nxt)
        zb = pd.DataFrame(np.array(sim), columns=zm.columns)
        zb = zb - zb.mean()
        try:
            vb = VAR(zb.values).fit(var.k_ar)
            boots.append(decompose(vb, zb))
        except Exception:
            continue
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) else None
    return {
        "freq": "quarterly", "n": int(len(z)), "var_lags": k, "rho": rho,
        "sample": [str(z.index[0]), str(z.index[-1])],
        "mean_yield_annual": float(yld_m.mean()),
        "share_rent_growth": s_rent,
        "share_expected_returns": s_ret,
        "share_riskfree": s_rf,
        "share_risk_premium": float(s_ret - s_rf),
        "ci_rent": [float(ci[0, 0]), float(ci[1, 0])] if ci is not None else None,
        "ci_returns": [float(ci[0, 1]), float(ci[1, 1])] if ci is not None else None,
        "ci_riskfree": [float(ci[0, 2]), float(ci[1, 2])] if ci is not None else None,
        "n_boot_ok": int(len(boots)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load_monthly()
    q = load_quarterly()
    log(f"monthly interpolated price obs: {panel.attrs['n_interp']}")

    ur = unit_root_battery(panel)
    ur.to_csv(os.path.join(OUT, "unit_roots.csv"), index=False)
    log("\n== unit roots ==\n" + ur.round(3).to_string(index=False))

    jo = johansen_table(panel)
    jo.to_csv(os.path.join(OUT, "johansen.csv"), index=False)
    log("\n== Johansen ==\n" + jo.round(3).to_string(index=False))

    results = {"n_interp": panel.attrs["n_interp"]}
    for y, x in [("lpr_A", "real_hibor"), ("lpr_ALL", "real_hibor"),
                 ("lpr_A", "real_gs10")]:
        b = uecm_bounds(panel, y, x)
        d_ = dols(panel, y, x)
        results[f"bounds_{y}_{x}"] = b
        results[f"dols_{y}_{x}"] = d_
        log(f"\nUECM bounds {y}~{x}: F={b['F_PSS']:.2f} (p_upper≈{b['F_p_upper']:.3f}) "
            f"EC={b['ec_coef']:.4f} (t={b['ec_t']:.2f}) "
            f"LR semi-elast={b['longrun_semi_elasticity']:.4f} "
            f"half-life={b['half_life_months']:.1f}m")
        log(f"DOLS {y}~{x}: beta={d_['beta']:.4f} (t={d_['t']:.2f}, n={d_['n']})")

    cs = campbell_shiller(q, panel)
    results["campbell_shiller"] = cs
    log(f"\nCampbell-Shiller (quarterly {cs['sample']}, VAR({cs['var_lags']}), "
        f"rho={cs['rho']:.4f}, n={cs['n']}):"
        f"\n  rent-growth share      = {cs['share_rent_growth']:.3f} {cs['ci_rent']}"
        f"\n  expected-return share  = {cs['share_expected_returns']:.3f} {cs['ci_returns']}"
        f"\n    of which real prime rate (BLR) = {cs['share_riskfree']:.3f} {cs['ci_riskfree']}"
        f"\n    of which premium over BLR      = {cs['share_risk_premium']:.3f}")

    with open(os.path.join(OUT, "longrun.json"), "w") as f:
        clean = json.loads(json.dumps(results, default=float),
                           parse_constant=lambda _: None)
        json.dump(clean, f, indent=2, allow_nan=False)
    with open(os.path.join(OUT, "longrun_log.txt"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
