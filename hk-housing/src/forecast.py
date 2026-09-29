"""
forecast.py — Pseudo out-of-sample forecasting horse race.

Rolling-origin evaluation of h-step-ahead forecasts of the log real price
index. Models:

  RW     : random walk (no-change) — the benchmark to beat.
  RWD    : random walk with rolling drift.
  AR     : AR(p) on log-returns, p by AIC, iterated.
  ECM    : error-correction forecast — Δŷ uses the lagged equilibrium error
           from a static OLS long-run relation re-estimated on each training
           window, iterated forward holding the fundamentals at their values
           at the origin. Only data up to the origin enter, so the caller must
           pass fundamentals as known at each date (the housing study passes
           income as published, not interpolated toward the next quarter).
  GBM    : gradient-boosted trees on lagged returns + fundamentals + ect,
           direct h-step forecast (sklearn HistGradientBoosting).

Inference: Diebold-Mariano (1995) test with the Harvey-Leybourne-Newbold
(1997) small-sample correction, squared-error loss, HAC variance with a
Bartlett kernel truncated at ceil(h/step) - 1 origin lags (h-step errors from
origins `step` months apart overlap by construction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, step: int = 1):
    """Diebold-Mariano with HLN small-sample correction. Loss: squared error.

    H0: equal predictive accuracy. Negative stat => model 1 better.
    `step`: spacing of forecast origins in months. h-step-ahead errors sampled
    every `step` months overlap up to lag ceil(h/step)-1 in origin units; the
    HAC variance uses a Bartlett kernel to that lag (positive semi-definite by
    construction, unlike the truncated rectangular kernel).
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    L = max(int(np.ceil(h / step)) - 1, 0)
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * w * cov
    var = max(var, 1e-12) / n
    dm = dbar / np.sqrt(var)
    he = L + 1  # effective forecast-horizon overlap in origin units
    hln = dm * np.sqrt(max(n + 1 - 2 * he + he * (he - 1) / n, 1) / n)
    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return float(hln), float(p)


def _fit_ar_forecast(y_tr: pd.Series, h: int, max_p: int = 6) -> float:
    """AIC-selected AR(p) on returns, iterated h steps; returns level forecast."""
    r = y_tr.diff().dropna()
    best_aic, best_p = np.inf, 1
    for p in range(1, max_p + 1):
        try:
            m = sm.tsa.ARIMA(r, order=(p, 0, 0)).fit()
            if m.aic < best_aic:
                best_aic, best_p = m.aic, p
        except Exception:
            continue
    m = sm.tsa.ARIMA(r, order=(best_p, 0, 0)).fit()
    fc = m.forecast(h)
    return float(y_tr.iloc[-1] + np.sum(fc))


def _fit_ecm_forecast(y_tr: pd.Series, X_tr: pd.DataFrame, h: int) -> float:
    """Static OLS long-run + short-run ECM estimated on the window, iterated h steps
    holding fundamentals fixed at their last observation."""
    data = pd.concat([y_tr, X_tr], axis=1).dropna()
    yv, Xv = data.iloc[:, 0], data.iloc[:, 1:]
    # Static OLS long-run fit; no DOLS lead/lag terms are included.
    lr = sm.OLS(yv, sm.add_constant(Xv)).fit()
    ect = yv - lr.predict(sm.add_constant(Xv))
    dy = yv.diff()
    Z = pd.DataFrame({"ect_l1": ect.shift(1), "dy_l1": dy.shift(1),
                      "dy_l2": dy.shift(2)})
    aligned = pd.concat([dy.rename("dy"), Z], axis=1).dropna()
    sr = sm.OLS(aligned["dy"], sm.add_constant(aligned.drop(columns="dy"))).fit()
    # iterate
    y_path = [float(yv.iloc[-1])]
    # Chronological order keeps [-1] as the most recent observed/forecast change.
    dy_hist = [float(dy.iloc[-2]), float(dy.iloc[-1])]
    x_last = Xv.iloc[-1]
    lr_part = float(lr.params.iloc[0] + (x_last * lr.params.iloc[1:]).sum())
    for _ in range(h):
        ect_now = y_path[-1] - lr_part
        dyf = float(sr.params["const"] + sr.params["ect_l1"] * ect_now
                    + sr.params["dy_l1"] * dy_hist[-1]
                    + sr.params["dy_l2"] * dy_hist[-2])
        y_path.append(y_path[-1] + dyf)
        dy_hist.append(dyf)
    return y_path[-1]


def _fit_gbm_forecast(y_tr: pd.Series, X_tr: pd.DataFrame, h: int,
                      n_lags: int = 6) -> float:
    """Direct h-step gradient boosting: y_{t+h}-y_t on lagged returns + X + ect."""
    dy = y_tr.diff()
    lr = sm.OLS(*[a for a in [pd.concat([y_tr, X_tr], axis=1).dropna().iloc[:, 0],
                              sm.add_constant(pd.concat([y_tr, X_tr], axis=1)
                                              .dropna().iloc[:, 1:])]]).fit()
    data = pd.concat([y_tr, X_tr], axis=1).dropna()
    ect = data.iloc[:, 0] - lr.predict(sm.add_constant(data.iloc[:, 1:]))
    feats = {f"dy_l{l}": dy.shift(l - 1) for l in range(1, n_lags + 1)}
    feats["ect"] = ect
    F = pd.concat([pd.DataFrame(feats), X_tr.diff()], axis=1)
    target = (y_tr.shift(-h) - y_tr).rename("tgt")
    train = pd.concat([target, F], axis=1).dropna()
    if len(train) < 60:
        return float(y_tr.iloc[-1])
    gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=300,
                                        learning_rate=0.05, random_state=0)
    gbm.fit(train.iloc[:, 1:], train["tgt"])
    x_now = F.iloc[[-1]]
    return float(y_tr.iloc[-1] + gbm.predict(x_now)[0])


def rolling_oos(y: pd.Series, X: pd.DataFrame, start_frac: float = 0.6,
                horizons=(1, 3, 12), step: int = 1):
    """Rolling-origin evaluation. Returns dict: horizon -> DataFrame of errors.

    Horizons are counted in rows, so the complete-case sample must be an
    unbroken monthly calendar; a gap would silently shift the targets.
    """
    data = pd.concat([y, X], axis=1).dropna()
    calendar = pd.date_range(data.index[0], data.index[-1], freq="ME")
    if len(calendar) != len(data) or not (data.index == calendar).all():
        raise ValueError("rolling_oos needs consecutive months after dropping "
                         "missing values")
    yv, Xv = data.iloc[:, 0], data.iloc[:, 1:]
    T = len(yv)
    t0 = int(T * start_frac)
    results = {h: [] for h in horizons}
    for t in range(t0, T - max(horizons), step):
        y_tr, X_tr = yv.iloc[:t + 1], Xv.iloc[:t + 1]
        drift = float(y_tr.diff().tail(60).mean())
        for h in horizons:
            actual = float(yv.iloc[t + h])
            row = {
                "origin": yv.index[t], "actual": actual,
                "RW": float(y_tr.iloc[-1]),
                "RWD": float(y_tr.iloc[-1] + h * drift),
                "AR": _fit_ar_forecast(y_tr, h),
                "ECM": _fit_ecm_forecast(y_tr, X_tr, h),
                "GBM": _fit_gbm_forecast(y_tr, X_tr, h),
            }
            results[h].append(row)
    return {h: pd.DataFrame(v).set_index("origin") for h, v in results.items()}


def oos_summary(oos: dict, models=("RW", "RWD", "AR", "ECM", "GBM"),
                step: int = 1):
    """RMSE table + DM tests vs RW for each horizon."""
    rows = []
    for h, df in oos.items():
        e = {m: (df[m] - df["actual"]).values for m in models}
        for m in models:
            rmse = float(np.sqrt(np.mean(e[m] ** 2)))
            if m == "RW":
                dm, p = 0.0, 1.0
            else:
                dm, p = dm_test(e[m], e["RW"], h=h, step=step)
            rows.append({"horizon": h, "model": m, "RMSE": rmse,
                         "rel_RW": rmse / float(np.sqrt(np.mean(e["RW"] ** 2))),
                         "DM_vs_RW": dm, "DM_p": p, "n": len(df)})
    return pd.DataFrame(rows)
