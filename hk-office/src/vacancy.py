"""
vacancy.py — The natural vacancy rate and office rent adjustment.

Rent adjustment equation (Hendershott 1996; Wheaton-Torto 1988):
    Δ ln R_t = λ (v* − v_{t−1}) + ε_t  =  α + β v_{t−1} + ε_t,
with λ = −β and v* = α / λ.  Estimated by grade (A, B, C, total) on annual
data: year-end vacancy rates (RVD private_office.xls, 1985+) against
Q4-on-Q4 real rent growth built from the *quarterly* rent index (alignment:
RVD measures vacancy at year-end).

Extensions:
  (i)   delta-method and pairs-bootstrap CIs for v*;
  (ii)  asymmetric adjustment: λ⁺ (v above v*) vs λ⁻ (v below v*), grid search
        over v*, with a wild-bootstrap test of λ⁺ = λ⁻ against the symmetric
        model with v* free;
  (iii) time-varying v*: local-level state-space model
            Δ ln R_t = λ (v*_t − v_{t−1}) + ε_t,   v*_t = v*_{t−1} + η_t,
        two-step: λ fixed at the OLS value, then Kalman MLE of the local level
        (statsmodels UnobservedComponents, several starting values), with a
        boundary likelihood-ratio test of a constant v* (var(η) = 0);
  (iv)  demand-shift controls (real GDP growth) to check robustness of v*.

Outputs: output/vacancy.json, output/vstar_path.csv, output/vacancy_log.txt
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
LOG: list[str] = []


def log(msg: str):
    LOG.append(str(msg))
    print(msg)


def build_annual() -> pd.DataFrame:
    ann = pd.read_csv(os.path.join(PRO, "annual_stockflow.csv"), index_col=0)
    q = pd.read_csv(os.path.join(PRO, "quarterly_indices.csv"), index_col=0)
    q.index = pd.PeriodIndex(q.index, freq="Q")
    q4 = q[q.index.quarter == 4].copy()
    q4.index = q4.index.year
    out = pd.DataFrame(index=ann.index)
    for g in ("A", "B", "C", "TOT"):
        gq = "ALL" if g == "TOT" else g
        # Q4 real rent (deflate by Q4 CPI), growth Dec-on-Dec
        rr = np.log(q4[f"rt_{gq}"] / q4["cpi"])
        out[f"dlr_{g}"] = rr.diff()
        out[f"vac_{g}"] = ann[f"vacrate_{g}"]
    out["dgdp"] = np.log(ann["gdp_real"]).diff()
    return out


# ----------------------------------------------------------- baseline OLS
def baseline(df: pd.DataFrame, g: str, n_boot: int = 5000, seed: int = 7) -> dict:
    d = pd.DataFrame({
        "y": df[f"dlr_{g}"],
        "vlag": df[f"vac_{g}"].shift(1),
    }).dropna()
    X = sm.add_constant(d["vlag"])
    res = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    a, b = res.params["const"], res.params["vlag"]
    lam = -b
    vstar = a / lam if lam > 0 else np.nan
    # delta method on v* = a / (-b)
    V = res.cov_params().values
    grad = np.array([1.0 / lam, a / lam**2])
    se_v = float(np.sqrt(grad @ V @ grad)) if lam > 0 else np.nan
    # pairs bootstrap
    rng = np.random.default_rng(seed)
    vs = []
    arr = d.values
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        yb, xb = arr[idx, 0], arr[idx, 1]
        Xb = np.column_stack([np.ones(len(xb)), xb])
        try:
            cb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            if cb[1] < 0:
                vs.append(cb[0] / -cb[1])
        except np.linalg.LinAlgError:
            continue
    vs = np.array(vs)
    ci = np.percentile(vs, [2.5, 97.5]) if len(vs) > 100 else [np.nan, np.nan]
    # robustness: control for contemporaneous GDP growth; v* evaluated at the
    # sample-mean GDP growth path (v* = rent-growth-neutral vacancy given
    # average demand growth), not at zero growth
    d2 = pd.DataFrame({"y": df[f"dlr_{g}"], "vlag": df[f"vac_{g}"].shift(1),
                       "dgdp": df["dgdp"]}).dropna()
    r2 = sm.OLS(d2["y"], sm.add_constant(d2[["vlag", "dgdp"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2})
    vstar_ctrl = float((r2.params["const"]
                        + r2.params["dgdp"] * d2["dgdp"].mean())
                       / -r2.params["vlag"]) \
        if r2.params["vlag"] < 0 else np.nan
    return {
        "grade": g, "n": int(res.nobs),
        "first_year": int(d.index.min()), "last_year": int(d.index.max()),
        "lambda": float(lam), "lambda_t": float(-res.tvalues["vlag"]),
        "alpha": float(a), "r2": float(res.rsquared),
        "vstar": float(vstar), "vstar_se_delta": se_v,
        "vstar_ci_boot": [float(ci[0]), float(ci[1])],
        "boot_kept": int(len(vs)), "n_boot": n_boot,
        "vstar_gdp_control": vstar_ctrl,
        "gdp_coef": float(r2.params["dgdp"]), "gdp_t": float(r2.tvalues["dgdp"]),
    }


# ----------------------------------------------------------- asymmetry
def asymmetry(df: pd.DataFrame, g: str, n_boot: int = 2000, seed: int = 11) -> dict:
    """Δln R = λ⁺(v*−v)·1[v>v*] + λ⁻(v*−v)·1[v≤v*]; grid over v*.

    Under H0 (λ⁺ = λ⁻) the model is Δln R = α + βv with v* = α/−β still free,
    so the restricted fit is OLS on [1, v]. The unrestricted model nests every
    symmetric fit, so its SSR is the smaller of the grid optimum and the
    restricted SSR. The wild bootstrap draws from the restricted fit and
    re-estimates both models, threshold included, in every replication."""
    d = pd.DataFrame({"y": df[f"dlr_{g}"], "vlag": df[f"vac_{g}"].shift(1)}).dropna()
    y, v = d["y"].values, d["vlag"].values
    grid = np.linspace(np.quantile(v, 0.15), np.quantile(v, 0.85), 61)
    Xr = np.column_stack([np.ones(len(v)), v])

    def fit_restricted(yy):
        beta = np.linalg.lstsq(Xr, yy, rcond=None)[0]
        return beta, float(np.sum((yy - Xr @ beta) ** 2))

    def fit_threshold(yy):
        best = None
        for vs in grid:
            gap = vs - v
            above = (v > vs).astype(float)
            X = np.column_stack([gap * above, gap * (1 - above)])
            beta = np.linalg.lstsq(X, yy, rcond=None)[0]
            ssr = float(np.sum((yy - X @ beta) ** 2))
            if best is None or ssr < best[2]:
                best = (vs, beta, ssr)
        return best

    def f_stat(yy):
        ssr_r = fit_restricted(yy)[1]
        ssr_u = min(fit_threshold(yy)[2], ssr_r)
        return (ssr_r - ssr_u) / (ssr_u / (len(yy) - 3))

    vs_hat, (lam_p, lam_m), _ = fit_threshold(y)
    beta_r, _ = fit_restricted(y)
    F = f_stat(y)
    fitted = Xr @ beta_r
    resid = y - fitted
    rng = np.random.default_rng(seed)
    Fb = np.array([f_stat(fitted + resid * rng.choice([-1.0, 1.0], len(resid)))
                   for _ in range(n_boot)])
    return {
        "grade": g, "vstar_grid": float(vs_hat),
        "lambda_above": float(lam_p), "lambda_below": float(lam_m),
        "vstar_symmetric": float(beta_r[0] / -beta_r[1]) if beta_r[1] < 0 else None,
        "F_sym": float(F), "p_wild": float(np.mean(Fb >= F)), "n_boot": n_boot,
    }


# ----------------------------------------------------- time-varying v*
def _fit_local_level(values: np.ndarray):
    """Local-level fit from several starting values (highest likelihood kept)
    and the constrained fit with a constant level (var(η) = 0)."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    mod = UnobservedComponents(values, level="local level")
    var_y = float(np.nanvar(values))
    starts = [None] + [[var_y * (1 - r), var_y * r] for r in (1e-6, 0.01, 0.1, 0.5)]
    res = max((mod.fit(start_params=sp, disp=False, maxiter=1000) for sp in starts),
              key=lambda r: r.llf)
    res_const = mod.fit_constrained({"sigma2.level": 0.0}, disp=False, maxiter=1000)
    lr = 2 * (res.llf - res_const.llf)
    # optimizer noise leaves LRs of order 1e-7 at the boundary: treat as zero
    return res, res_const, lr if lr > 1e-6 else 0.0


def time_varying(df: pd.DataFrame, g: str, lam: float, n_boot: int = 499,
                 seed: int = 13) -> tuple[dict, pd.DataFrame]:
    """Profile-likelihood two-step: with λ fixed at the OLS estimate, the
    model Δln R_t = λ(v*_t − v_{t−1}) + ε, v*_t = v*_{t−1} + η reduces to a
    local-level model on the adjusted series y_t + λ·v_{t−1} = λ·v*_t + ε.
    (Free-λ MLE is unidentified at λ→0 on n≈40; the two-step pins λ to the
    rent-adjustment speed and lets the Kalman smoother allocate the rest.)

    The likelihood in var(η) can have an interior local mode as well as its
    maximum at the var(η) = 0 boundary, so the fit starts from the default and
    from several signal-to-noise ratios and keeps the highest likelihood. A
    constant v* (var(η) = 0) is tested by a likelihood ratio whose null
    distribution is simulated (parametric bootstrap: a constant level plus the
    fitted noise, refitted by both models), because the LR for a random-walk
    variance has a point mass at zero and no standard limit. A year missing
    inside the sample stays in the model as a missing observation."""
    d = pd.DataFrame({"y": df[f"dlr_{g}"], "vlag": df[f"vac_{g}"].shift(1)})
    years = d.dropna().index
    d = d.reindex(range(years.min(), years.max() + 1))
    yadj = d["y"] + lam * d["vlag"]
    values = yadj.to_numpy(float)
    res, res_const, lr = _fit_local_level(values)
    # null draws: the constrained fit's constant level plus N(0, its noise)
    rng = np.random.default_rng(seed)
    level0 = float(res_const.smoothed_state[0][0])
    sd0 = float(np.sqrt(res_const.params[0]))
    observed = ~np.isnan(values)
    lr_null = []
    for _ in range(n_boot):
        sim = np.full(len(values), np.nan)
        sim[observed] = level0 + sd0 * rng.standard_normal(observed.sum())
        lr_null.append(_fit_local_level(sim)[2])
    lr_null = np.array(lr_null)
    lvl = res.smoothed_state[0] / lam
    lvl_se = np.sqrt(res.smoothed_state_cov[0, 0]) / lam
    path = pd.DataFrame({"year": d.index, "vstar_smoothed": lvl,
                         "se": lvl_se}).set_index("year")
    snr = float(res.params[1] / res.params[0]) if res.params[0] > 0 else None
    info = {
        "grade": g, "lambda_fixed": float(lam),
        "n": int(yadj.notna().sum()), "first_year": int(d.index[0]),
        "last_year": int(d.index[-1]),
        "var_eps": float(res.params[0]), "var_eta": float(res.params[1]),
        "signal_noise": snr, "loglike": float(res.llf),
        "loglike_constant": float(res_const.llf),
        "lr_constant": float(lr),
        "p_constant": float((1 + np.sum(lr_null >= lr)) / (1 + n_boot)),
        "lr_null_share_zero": float(np.mean(lr_null == 0.0)), "n_boot": n_boot,
        "vstar_first5_mean": float(lvl[:5].mean()),
        "vstar_last5_mean": float(lvl[-5:].mean()),
        "vstar_2019": float(path.loc[2019, "vstar_smoothed"]) if 2019 in path.index else None,
        "vstar_end": float(lvl[-1]),
        "vstar_se_mean": float(lvl_se.mean()),
        "converged": bool(res.mle_retvals.get("converged", True)),
    }
    return info, path


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build_annual()
    results = {"baseline": [], "asymmetry": [], "time_varying": []}
    paths = {}
    for g in ("A", "B", "C", "TOT"):
        b = baseline(df, g)
        results["baseline"].append(b)
        log(f"[{g}] λ={b['lambda']:.3f} (t={b['lambda_t']:.2f})  "
            f"v*={b['vstar']:.3%} ±{b['vstar_se_delta']:.3%} "
            f"boot95={[round(x, 4) for x in b['vstar_ci_boot']]} "
            f"(n={b['n']}, R²={b['r2']:.2f}; v* w/GDP ctrl={b['vstar_gdp_control']:.3%})")
        a = asymmetry(df, g)
        results["asymmetry"].append(a)
        log(f"    asym: λ⁺={a['lambda_above']:.3f} λ⁻={a['lambda_below']:.3f} "
            f"(F={a['F_sym']:.2f}, wild-p={a['p_wild']:.3f}, v*grid={a['vstar_grid']:.3%})")
        try:
            tv, path = time_varying(df, g, b["lambda"])
            results["time_varying"].append(tv)
            paths[g] = path
            log(f"    TV v*: λ={tv['lambda_fixed']:.3f}  "
                f"v*(first5)={tv['vstar_first5_mean']:.3%} → v*(last5)={tv['vstar_last5_mean']:.3%} "
                f"(end={tv['vstar_end']:.3%}; var(η)={tv['var_eta']:.2e}, "
                f"LR vs constant={tv['lr_constant']:.2f}, p={tv['p_constant']:.3f})")
        except Exception as e:
            log(f"    TV v* failed for {g}: {e}")

    columns = {f"vstar_{g}": p["vstar_smoothed"] for g, p in paths.items()}
    columns.update({f"se_{g}": p["se"] for g, p in paths.items()})
    allp = pd.concat(columns, axis=1).sort_index()
    allp = allp.reindex(range(allp.index.min(), allp.index.max() + 1))
    allp.index.name = "year"
    vac = df[[c for c in df.columns if c.startswith("vac_")]]
    allp.join(vac).to_csv(os.path.join(OUT, "vstar_path.csv"))

    with open(os.path.join(OUT, "vacancy.json"), "w") as f:
        clean = json.loads(json.dumps(results, default=float),
                           parse_constant=lambda _: None)
        json.dump(clean, f, indent=2, allow_nan=False)
    with open(os.path.join(OUT, "vacancy_log.txt"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
