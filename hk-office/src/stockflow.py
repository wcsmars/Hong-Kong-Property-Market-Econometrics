"""
stockflow.py — Stock-flow supply, office demand, and a stock-flow
absorption simulation (DiPasquale-Wheaton space-market logic, no price block).

(a) Supply: completions respond to lagged real capital values
        ln comp_t = a + b·ln(P/CPI)_{t−k} + c·real_rate_{t−k} + e,
    lag k (construction lag) selected over {2,3,4} by AIC on a common sample,
    then re-estimated on its full sample; elasticity b.
(b) Demand: occupied stock OS_t = S_t (1 − v_t);
        ln OS_t = α + β_y ln GDP_t + β_r ln(R/CPI)_t + u_t   (long run),
    estimated by DOLS, with an ECM for short-run absorption.
(c) Stock-flow simulation, 2026-2035, from end-2025 initial conditions.
    Completions are exogenous scenarios (30%, 100% or 150% of the 2021-25
    mean), because (a) finds no positive price response. Real rents adjust to
    the gap between vacancy and v* at vacancy.py's total-market (λ, v*).
    Occupied stock moves toward the demand schedule at speed κ (the ECM speed,
    floored at 0.10), under three GDP growth scenarios. Prices are not
    simulated. Output: vacancy paths and the year vacancy first reaches v*.

Outputs: output/stockflow.json, output/simulation_paths.csv,
         output/stockflow_log.txt
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
LOG: list[str] = []
KAPPA_FLOOR = 0.1   # minimum demand adjustment speed used in the simulation


def log(msg: str):
    LOG.append(str(msg))
    print(msg)


def build() -> pd.DataFrame:
    ann = pd.read_csv(os.path.join(PRO, "annual_stockflow.csv"), index_col=0)
    df = pd.DataFrame(index=ann.index)
    df["comp"] = ann["comp_TOT"]
    df["stock"] = ann["stock_TOT"]
    df["vac"] = ann["vacrate_TOT"]
    df["takeup"] = ann["takeup_TOT"]
    df["lrealp"] = np.log(ann["idx_px_ALL"] / ann["cpi"])
    df["lrealr"] = np.log(ann["idx_rt_ALL"] / ann["cpi"])
    df["lgdp"] = np.log(ann["gdp_real"])
    df["infl"] = 100.0 * (ann["cpi"] / ann["cpi"].shift(1) - 1.0)
    df["real_blr"] = ann["blr"] - df["infl"]
    df["los"] = np.log(df["stock"] * (1.0 - df["vac"]))
    df["lcomp"] = np.log(df["comp"])
    return df


# ----------------------------------------------------------------- supply
def supply_equation(df: pd.DataFrame, lags: tuple[int, ...] = (2, 3, 4)) -> dict:
    def data(k):
        return pd.DataFrame({
            "lcomp": df["lcomp"],
            "lrealp_lag": df["lrealp"].shift(k),
            "real_blr_lag": df["real_blr"].shift(k),
        }).dropna()

    def fit(d):
        return sm.OLS(d["lcomp"], sm.add_constant(d[["lrealp_lag", "real_blr_lag"]])
                      ).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    # AIC is only comparable on one sample: use the years every lag covers
    common = data(max(lags)).index
    aic = {k: float(fit(data(k).loc[common]).aic) for k in lags}
    k = min(aic, key=aic.get)
    d = data(k)
    res = fit(d)
    return {
        "lag_years": k, "aic_common_sample": aic, "n": int(res.nobs),
        "price_elasticity": float(res.params["lrealp_lag"]),
        "price_t": float(res.tvalues["lrealp_lag"]),
        "rate_semi": float(res.params["real_blr_lag"]),
        "rate_t": float(res.tvalues["real_blr_lag"]),
        "r2": float(res.rsquared),
        "const": float(res.params["const"]),
        "mean_lcomp": float(d["lcomp"].mean()),
        "mean_lrealp": float(d["lrealp_lag"].mean()),
    }


# ----------------------------------------------------------------- demand
def demand_dols(df: pd.DataFrame, leads: int = 2) -> dict:
    d = df[["los", "lgdp", "lrealr"]].dropna().copy()
    X = d[["lgdp", "lrealr"]].copy()
    for col in ("lgdp", "lrealr"):
        dx = d[col].diff()
        for j in range(-leads, leads + 1):
            X[f"d{col}_{j}"] = dx.shift(-j)
    data = pd.concat([d["los"].rename("y"), X], axis=1).dropna()
    res = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))
                 ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    u = d["los"] - (res.params["const"] + res.params["lgdp"] * d["lgdp"]
                    + res.params["lrealr"] * d["lrealr"])
    # ECM: Δlos on lagged disequilibrium + Δlgdp
    e = pd.DataFrame({"dlos": d["los"].diff(), "ulag": u.shift(1),
                      "dlgdp": d["lgdp"].diff()}).dropna()
    ecm = sm.OLS(e["dlos"], sm.add_constant(e[["ulag", "dlgdp"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2})
    return {
        "n": int(res.nobs),
        "income_elasticity": float(res.params["lgdp"]),
        "income_t": float(res.tvalues["lgdp"]),
        "rent_elasticity": float(res.params["lrealr"]),
        "rent_t": float(res.tvalues["lrealr"]),
        "const": float(res.params["const"]),
        "ecm_coef": float(ecm.params["ulag"]), "ecm_t": float(ecm.tvalues["ulag"]),
        "ecm_dlgdp": float(ecm.params["dlgdp"]),
    }


# ------------------------------------------------------------- simulation
def simulate(df: pd.DataFrame, sup: dict, dem: dict, q2: dict,
             horizon: int = 10, adj_speed: float | None = None,
             demand_recovery_years: int = 0) -> pd.DataFrame:
    """Stock-flow forward simulation from end-2025 conditions.

    adj_speed: override for the demand partial-adjustment speed κ (default:
    −ECM coefficient, floored at KAPPA_FLOOR). demand_recovery_years > 0 makes the
    absorbed 2025 demand residual TRANSITORY: the gap between end-2025
    occupancy and the historical long-run schedule decays linearly to zero
    over that many years instead of being treated as a permanent level shift.
    """
    lam = q2["lambda"]; vstar = q2["vstar"]
    y0 = int(df["vac"].dropna().index.max())   # last complete stock-flow year
    S = float(df.loc[y0, "stock"])
    v = float(df.loc[y0, "vac"])
    lrr = float(df.loc[y0, "lrealr"])
    OS = S * (1 - v)
    # GDP level at y0: extrapolate the last World Bank observation at the
    # base-scenario growth rate for any missing years
    lgdp_obs = df["lgdp"].dropna()
    lgdp0 = float(lgdp_obs.iloc[-1]) + 0.025 * (y0 - int(lgdp_obs.index.max()))
    # demand intercept from the HISTORICAL schedule, and the end-2025 residual
    # (how far occupancy sits below that schedule)
    resid_2025 = float(np.log(OS) - (dem["const"]
                                     + dem["income_elasticity"] * lgdp0
                                     + dem["rent_elasticity"] * lrr))
    q2["resid_2025"] = resid_2025
    # baseline: residual is permanent (absorbed into the intercept)
    c_dem = dem["const"] + resid_2025

    # recent completions for the pipeline floor (committed projects deliver);
    # dropna() before tail(5): the frame's final row (index-year 2026) has no
    # completions data and would silently shrink the window to four years
    comp_recent = float(df["comp"].dropna().tail(5).mean())

    # Completions are EXOGENOUS scenarios: the estimated supply equation
    # (negative price "elasticity") shows HK office completions are driven by
    # land-disposal policy and 4-year construction lags, not by prices —
    # endogenising them on price would inject a perverse feedback.
    scen_gdp = {"weak": 0.01, "base": 0.025, "strong": 0.04}
    scen_pipe = {"halt": 0.3, "base": 1.0, "heavy": 1.5}

    rows = []
    for (gn, gg), (pn, pmult) in itertools.product(
            scen_gdp.items(), scen_pipe.items()):
        S_, v_, lrr_ = S, v, lrr
        OS_cur = OS
        lgdp_ = lgdp0
        norm_year = None
        for h in range(1, horizon + 1):
            yr = y0 + h
            lgdp_ += np.log(1 + gg)
            # rent adjustment (vacancy.py estimates): real rents respond to excess vacancy
            lrr_ += lam * (vstar - v_)
            comp_new = comp_recent * pmult
            S_ = S_ + comp_new
            # demand: long-run target + partial adjustment at speed κ;
            # under a transitory shock, the 2025 residual decays to zero
            if demand_recovery_years > 0:
                share = max(0.0, 1.0 - h / demand_recovery_years)
                c_now = dem["const"] + resid_2025 * share
            else:
                c_now = c_dem
            os_target = np.exp(c_now + dem["income_elasticity"] * lgdp_
                               + dem["rent_elasticity"] * lrr_)
            adj = adj_speed if adj_speed is not None \
                else min(max(-dem["ecm_coef"], KAPPA_FLOOR), 1.0)
            OS_cur = OS_cur * (os_target / OS_cur) ** adj
            OS_cur = min(OS_cur, 0.98 * S_)
            v_ = 1.0 - OS_cur / S_
            if norm_year is None and v_ <= vstar:
                norm_year = yr
            rows.append({"scenario": f"gdp_{gn}|pipe_{pn}",
                         "year": yr, "vacancy": v_, "stock": S_,
                         "real_rent_log": lrr_, "completions": comp_new,
                         "occupied": OS_cur, "norm_year": norm_year})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build()
    sup = supply_equation(df)
    log(f"supply: lag={sup['lag_years']}y elasticity={sup['price_elasticity']:.2f} "
        f"(t={sup['price_t']:.2f}) rate={sup['rate_semi']:.3f} (t={sup['rate_t']:.2f}) "
        f"R²={sup['r2']:.2f} n={sup['n']}")
    dem = demand_dols(df)
    log(f"demand (DOLS): income elast={dem['income_elasticity']:.3f} "
        f"(t={dem['income_t']:.1f}), rent elast={dem['rent_elasticity']:.3f} "
        f"(t={dem['rent_t']:.1f}); ECM={dem['ecm_coef']:.3f} (t={dem['ecm_t']:.1f})")

    # rent-adjustment parameters from vacancy.py: total-market λ and v*
    with open(os.path.join(OUT, "vacancy.json")) as f:
        vac = json.load(f)
    q2b = [b for b in vac["baseline"] if b["grade"] == "TOT"][0]
    panel = pd.read_csv(os.path.join(PRO, "monthly_panel.csv"),
                        index_col=0, parse_dates=True)
    cap_anchor = float(panel["yld_A_mean"].dropna().tail(12).mean())
    q2 = {"lambda": q2b["lambda"], "vstar": q2b["vstar"],
          "cap_anchor": cap_anchor}
    log(f"sim params: λ={q2['lambda']:.2f}, v*={q2['vstar']:.3f}, "
        f"cap anchor (last-12m Grade A yield, reported only)={cap_anchor:.4f}")

    sim = simulate(df, sup, dem, q2)
    sim.to_csv(os.path.join(OUT, "simulation_paths.csv"), index=False)

    def summarize(s):
        return (s.groupby("scenario")
                .agg(vac_2030=("vacancy", lambda x: x.iloc[4]),
                     vac_end=("vacancy", "last"),
                     norm_year=("norm_year", "last"))
                .reset_index())

    summary = summarize(sim)
    log(f"\npipeline mean (last 5 observed years): "
        f"{float(df['comp'].dropna().tail(5).mean()):,.0f} m2/yr; "
        f"absorbed 2025 demand residual: {q2['resid_2025']:+.4f} log pts")
    log("\n== scenario summary (total market) ==\n"
        + summary.to_string(index=False,
                            formatters={"vac_2030": "{:.1%}".format,
                                        "vac_end": "{:.1%}".format}))

    # sensitivity 1: adjustment speed at the 95% upper bound of the ECM CI
    kappa_hi = float(-(dem["ecm_coef"]) + 1.96 * abs(dem["ecm_coef"] / dem["ecm_t"]))
    sim_k = simulate(df, sup, dem, dict(q2), adj_speed=kappa_hi)
    summ_k = summarize(sim_k)
    log(f"\n== κ-sensitivity (κ = {kappa_hi:.3f}, 95% upper bound) ==\n"
        + summ_k.to_string(index=False,
                           formatters={"vac_2030": "{:.1%}".format,
                                       "vac_end": "{:.1%}".format}))

    # sensitivity 2: transitory demand shock (2025 residual decays over 5y)
    sim_r = simulate(df, sup, dem, dict(q2), demand_recovery_years=5)
    summ_r = summarize(sim_r)
    log("\n== demand-recovery sensitivity (residual decays over 5y) ==\n"
        + summ_r.to_string(index=False,
                           formatters={"vac_2030": "{:.1%}".format,
                                       "vac_end": "{:.1%}".format}))

    results = {"supply": sup, "demand": dem, "sim_params": q2,
               "pipeline_mean_5y": float(df["comp"].dropna().tail(5).mean()),
               "pipeline_years": [int(df["comp"].dropna().tail(5).index.min()),
                                  int(df["comp"].dropna().tail(5).index.max())],
               "kappa_floor": KAPPA_FLOOR,
               "resid_2025_logpts": q2["resid_2025"],
               "scenarios": summary.to_dict("records"),
               "kappa_upper": kappa_hi,
               "scenarios_kappa_upper": summ_k.to_dict("records"),
               "scenarios_demand_recovery": summ_r.to_dict("records")}
    with open(os.path.join(OUT, "stockflow.json"), "w") as f:
        clean = json.loads(json.dumps(results, default=float),
                           parse_constant=lambda _: None)
        json.dump(clean, f, indent=2, allow_nan=False)
    with open(os.path.join(OUT, "stockflow_log.txt"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
