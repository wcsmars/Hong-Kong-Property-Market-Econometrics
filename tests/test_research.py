"""Offline regression checks: synthetic inputs and bundled result snapshots.

Run from the repository root: python -m unittest discover -s tests -v
These checks do not certify the empirical designs or require raw datasets.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("hk-housing", "hk-office")


def load_module(project: str, name: str):
    # Numba stores the module name in its disk cache. Keep PSY's usual import
    # name so this suite can share cached kernels with src/test_psy.py.
    key = "psy" if name == "psy" else f"public_{project.replace('-', '_')}_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            key, ROOT / project / "src" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


class RecursiveADFTests(unittest.TestCase):
    def test_adf_matches_statsmodels_for_full_and_partial_windows(self):
        psy = load_module(PROJECTS[0], "psy")
        y = np.cumsum(np.random.default_rng(1729).normal(size=180))
        for lag in (0, 1, 4):
            for start, end in ((0, 179), (31, 128)):
                with self.subTest(lag=lag, start=start, end=end):
                    expected = adfuller(
                        y[start:end + 1], maxlag=lag,
                        regression="c", autolag=None,
                    )[0]
                    actual = psy.adf_stat(y, k=lag, s=start, e=end)
                    self.assertAlmostEqual(actual, expected, places=8)

    def test_sadf_uses_requested_lag(self):
        psy = load_module(PROJECTS[0], "psy")
        y = np.cumsum(np.random.default_rng(314).normal(size=120))
        w0 = 30
        for lag in (0, 1, 2):
            with self.subTest(lag=lag):
                expected = max(
                    adfuller(y[:end + 1], maxlag=lag, regression="c", autolag=None)[0]
                    for end in range(w0 - 1, len(y))
                )
                self.assertAlmostEqual(psy.sadf(y, k=lag, w0=w0), expected, places=8)


class BoundsTestTests(unittest.TestCase):
    def test_critical_values_and_p_values_count_only_regressors(self):
        # PSS (2001) case III, 5%: (I(0), I(1)) bounds by number of regressors.
        published = {1: (4.92, 5.72), 2: (3.80, 4.81), 3: (3.23, 4.32)}
        for project in PROJECTS:
            module = load_module(project, "longrun")
            for k, bounds in published.items():
                with self.subTest(project=project, k=k):
                    crit, _ = module.pss_bounds(1.0, k=k)
                    self.assertAlmostEqual(crit.loc[95, "lower"], bounds[0], places=2)
                    self.assertAlmostEqual(crit.loc[95, "upper"], bounds[1], places=2)
                    # A statistic at the 5% critical value has a p-value near 5%.
                    _, pvals = module.pss_bounds(crit.loc[95, "upper"], k=k)
                    self.assertAlmostEqual(pvals["upper"], 0.05, delta=0.005)

    def test_housing_bounds_test_keeps_every_regressor(self):
        # AIC drops the irrelevant first regressor here, so mapping the
        # selected orders onto the leading columns would test "noise" alone.
        rng = np.random.default_rng(0)
        n = 200
        idx = pd.date_range("2000-01-31", periods=n, freq="ME")
        signal = np.cumsum(rng.normal(size=n))
        noise = np.cumsum(rng.normal(size=n))
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.5 * u[t - 1] + rng.normal(scale=0.5)
        y = pd.Series(1.0 + 0.8 * signal + u, index=idx, name="y")
        X = pd.DataFrame({"noise": noise, "signal": signal}, index=idx)
        longrun = load_module(PROJECTS[0], "longrun")
        unconstrained = longrun.ardl_select_order(y, 3, X, 3, trend="c", ic="aic")
        self.assertIsNone(unconstrained.aic.iloc[0][1]["noise"])

        result = longrun.ardl_bounds(y, X, maxlag=3)
        levels = [name for name in result["uecm_res"].model.exog_names
                  if name.endswith(".L1") and not name.startswith("D.")]
        self.assertEqual(levels, ["y.L1", "noise.L1", "signal.L1"])
        self.assertEqual(result["k"], 2)
        self.assertEqual(len(result["ardl_order"]), 3)
        self.assertAlmostEqual(result["bounds_crit"].loc[95, "upper"], 4.81, places=2)


class DistrictDispersionTests(unittest.TestCase):
    def test_balanced_panel_is_not_moved_by_a_late_entrant(self):
        breaks = load_module(PROJECTS[1], "breaks")
        rng = np.random.default_rng(5)
        idx = pd.date_range("2013-01-31", periods=144, freq="ME")
        rel = pd.DataFrame({
            "ar_A_North": 0.2 + 0.01 * rng.normal(size=144),
            "ar_A_South": -0.2 + 0.01 * rng.normal(size=144),
            "ar_A_East": -1.1 + 0.01 * rng.normal(size=144),
        }, index=idx)
        rel.loc[:"2016-12-31", "ar_A_East"] = np.nan  # enters in 2017

        _, observed = breaks.sigma_split(rel)
        _, balanced = breaks.sigma_split(rel.dropna(how="any"))
        self.assertGreater(observed["post2019_mean"] - observed["pre2019_mean"], 0.2)
        self.assertAlmostEqual(balanced["pre2019_mean"], balanced["post2019_mean"], delta=0.01)
        self.assertEqual(balanced["start"], "2017-01-31")
        self.assertEqual(balanced["districts"], ["Central", "North", "South", "East"])


class ForecastRecurrenceTests(unittest.TestCase):
    def test_ecm_uses_latest_change_as_first_lag(self):
        # Distinct final changes (4 and 3) make reversed lag order observable.
        # Fit coefficients are fixed, so this checks iteration rather than
        # comparing two copies of the same estimation routine.
        idx = pd.date_range("2000-01-31", periods=5, freq="ME")
        y = pd.Series([2.0, 3.0, 5.0, 8.0, 12.0], index=idx, name="y")
        x = pd.DataFrame({"x": [1., 2., 3., 4., 5.]}, index=idx)
        long_params = pd.Series({"const": 0.5, "x": 0.4})
        short_params = pd.Series({
            "const": 0.1, "ect_l1": -0.2, "dy_l1": 0.6, "dy_l2": -0.1,
        })

        # Independent scalar recurrence: level=12, equilibrium=.5+.4*5.
        level, latest_change, preceding_change = 12.0, 4.0, 3.0
        expected = {}
        for horizon in range(1, 5):
            next_change = (
                0.1 - 0.2 * (level - 2.5)
                + 0.6 * latest_change - 0.1 * preceding_change
            )
            level += next_change
            preceding_change, latest_change = latest_change, next_change
            expected[horizon] = level
        self.assertAlmostEqual(expected[1], 12.3)
        self.assertAlmostEqual(expected[2], 10.22)

        for project in PROJECTS:
            module = load_module(project, "forecast")
            for horizon in (1, 2, 4):
                long_fit = SimpleNamespace(
                    params=long_params,
                    predict=lambda design: design @ long_params,
                )
                short_fit = SimpleNamespace(params=short_params)
                fitted_models = [
                    SimpleNamespace(fit=lambda: long_fit),
                    SimpleNamespace(fit=lambda: short_fit),
                ]
                with self.subTest(project=project, horizon=horizon):
                    with patch.object(module.sm, "OLS", side_effect=fitted_models):
                        actual = module._fit_ecm_forecast(y, x, horizon)
                    self.assertAlmostEqual(actual, expected[horizon], places=10)


class CalendarValidationTests(unittest.TestCase):
    @staticmethod
    def worksheet(months, all_column):
        data = pd.DataFrame(np.nan, index=range(len(months)), columns=range(30))
        data = data.astype(object)
        data.iloc[0, 1] = "2004"  # The year can be text in RVD workbooks.
        for row, month in enumerate(months):
            data.iloc[row, 5] = month
            data.iloc[row, all_column] = f"({100 + row}.0)"
        return data

    def parse(self, project, months):
        module = load_module(project, "pipeline")
        if project == PROJECTS[0]:
            parser, all_column = module.parse_rvd_monthly, 29
        else:
            parser, all_column = module.parse_grade_monthly, 17
        with patch.object(module.pd, "read_excel", return_value=self.worksheet(months, all_column)):
            return parser("synthetic.xls", "px")

    def test_accepts_complete_calendar_and_text_year(self):
        for project in PROJECTS:
            with self.subTest(project=project):
                result = self.parse(project, [1, 2, 3])
                expected = pd.date_range("2004-01-31", periods=3, freq="ME")
                pd.testing.assert_index_equal(result.index, expected, check_names=False)
                np.testing.assert_array_equal(result["px_ALL"], [100., 101., 102.])

    def test_rejects_duplicate_months(self):
        for project in PROJECTS:
            with self.subTest(project=project):
                with self.assertRaisesRegex(ValueError, "duplicated"):
                    self.parse(project, [1, 2, 2])

    def test_rejects_missing_months(self):
        for project in PROJECTS:
            with self.subTest(project=project):
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.parse(project, [1, 3])


class SavedForecastSnapshotTests(unittest.TestCase):
    def test_saved_summaries_agree_with_forecast_rows(self):
        for project in PROJECTS:
            output = ROOT / project / "output"
            summary = pd.read_csv(output / "oos_summary.csv")
            self.assertFalse(summary.duplicated(["horizon", "model"]).any())
            self.assertEqual(set(summary["horizon"]), {1, 3, 12})
            forecasts = {}
            for horizon in (1, 3, 12):
                rows = pd.read_csv(output / f"oos_h{horizon}.csv", index_col=0)
                self.assertFalse(rows.index.duplicated().any())
                forecasts[horizon] = rows
                baseline = np.sqrt(np.mean((rows["RW"] - rows["actual"]) ** 2))
                for _, saved in summary.loc[summary["horizon"] == horizon].iterrows():
                    with self.subTest(project=project, horizon=horizon, model=saved["model"]):
                        error = rows[saved["model"]] - rows["actual"]
                        self.assertTrue(np.isfinite(error).all())
                        rmse = np.sqrt(np.mean(error ** 2))
                        self.assertAlmostEqual(rmse, saved["RMSE"], places=12)
                        self.assertAlmostEqual(rmse / baseline, saved["rel_RW"], places=12)
                        self.assertEqual(len(rows), saved["n"])

            # Also check the stored DM statistics use the documented spacing.
            module = load_module(project, "forecast")
            actual = module.oos_summary(forecasts, step=2 if project == PROJECTS[0] else 1)
            keys = ["horizon", "model"]
            actual = actual.set_index(keys).sort_index()
            expected = summary.set_index(keys).sort_index()
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-9)


class LaggedPSYTests(unittest.TestCase):
    def test_one_lag_bsadf_matches_statsmodels_on_every_window(self):
        # Every published bubble statistic uses one lag; check the general
        # path and the prefix-sum path used by the simulations window by window.
        psy = load_module(PROJECTS[0], "psy")
        y = np.cumsum(np.random.default_rng(2718).normal(size=48))
        w0 = 16
        general = psy.bsadf(y, k=1, w0=w0)
        fast = np.full(len(y), -np.inf)
        psy._bsadf_fast(y, w0, 1, fast)
        for end in range(w0 - 1, len(y)):
            expected = max(
                adfuller(y[start:end + 1], maxlag=1, regression="c", autolag=None)[0]
                for start in range(0, end - w0 + 2)
            )
            with self.subTest(end=end):
                self.assertAlmostEqual(general[end], expected, places=8)
                self.assertAlmostEqual(fast[end], expected, places=8)

    def test_null_simulation_and_bootstrap_use_the_requested_dynamics(self):
        psy = load_module(PROJECTS[0], "psy")
        # A zero AR coefficient reproduces the iid random-walk null exactly.
        iid = psy.mc_critical_values(60, w0=20, n_reps=40, seed=3, k=1)
        ar0 = psy.mc_critical_values(60, w0=20, n_reps=40, seed=3, k=1, psi=[0.0])
        np.testing.assert_array_equal(iid["null_gsadf"], ar0["null_gsadf"])
        # The bootstrap's null autoregression recovers the change dynamics.
        rng = np.random.default_rng(12)
        dy = np.zeros(3000)
        for t in range(1, len(dy)):
            dy[t] = 0.6 * dy[t - 1] + rng.normal()
        psi, resid = psy.fit_null_ar(np.concatenate([[0.0], np.cumsum(dy)]), 1)
        self.assertAlmostEqual(psi[0], 0.6, delta=0.05)
        self.assertAlmostEqual(resid.mean(), 0.0, places=10)

    def test_null_draws_follow_the_ar_recursion(self):
        # Rebuild draws independently: numba's seeded normals equal NumPy's
        # legacy RandomState, so each simulated GSADF can be recomputed.
        psy = load_module(PROJECTS[0], "psy")
        T, w0, seed = 60, 20, 11
        null = psy.mc_critical_values(T, w0=w0, n_reps=3, seed=seed, k=1, psi=[0.5])
        for rep in range(3):
            eps = np.random.RandomState(seed + rep).standard_normal(T)
            changes = np.zeros(T)
            for t in range(T):
                changes[t] = eps[t] + (0.5 * changes[t - 1] if t else 0.0)
            expected = psy.gsadf(np.cumsum(changes), k=1, w0=w0)
            self.assertAlmostEqual(null["null_gsadf"][rep], expected, places=9)

    def test_bootstrap_draws_follow_the_ar_recursion(self):
        psy = load_module(PROJECTS[0], "psy")
        y = np.cumsum(0.3 * np.random.default_rng(1).normal(size=80))
        boot = psy.wild_bootstrap_gsadf(y, k=1, w0=20, n_reps=3, seed=5)
        psi, resid = psy.fit_null_ar(y, 1)
        observed = np.diff(y)
        for rep in range(3):
            weights = np.random.RandomState(5 + 7919 * rep).standard_normal(len(resid))
            changes = np.empty(len(observed))
            changes[0] = observed[0]           # first change kept, then the recursion
            for j in range(len(resid)):
                changes[1 + j] = psi[0] * changes[j] + weights[j] * resid[j]
            expected = psy.gsadf(np.concatenate([[0.0], np.cumsum(changes)]), k=1, w0=20)
            self.assertAlmostEqual(boot["null_gsadf"][rep], expected, places=9)


class PlaceboInferenceTests(unittest.TestCase):
    def test_placebo_calendars_keep_spacing_and_avoid_real_events(self):
        policy = load_module(PROJECTS[0], "policy")
        idx = pd.date_range("2000-01-31", periods=240, freq="ME")
        events = [pd.Timestamp("2010-03-31"), pd.Timestamp("2010-07-31")]
        calendars = policy.placebo_calendars(idx, events, events, window=24)
        self.assertTrue(calendars)
        for positions in calendars:
            self.assertEqual(positions[1] - positions[0], 4)
            for pos in positions:
                months_away = min(abs((idx[pos].year - e.year) * 12 + idx[pos].month - e.month)
                                  for e in events)
                self.assertGreater(months_away, 24)

    def test_placebo_betas_equal_regressions_with_the_placebo_dummy(self):
        policy = load_module(PROJECTS[0], "policy")
        rng = np.random.default_rng(4)
        W = np.column_stack([np.ones(120), rng.normal(size=(120, 3))])
        yv = W @ np.array([0.2, 1.0, -0.5, 0.3]) + rng.normal(size=120)
        calendars = [np.array([7]), np.array([30, 34, 60]), np.array([100, 101])]
        betas = policy.placebo_betas(W, yv, calendars)
        for beta, positions in zip(betas, calendars):
            dummy = np.zeros(120)
            dummy[positions] = 1.0
            fit = sm.OLS(yv, np.column_stack([W, dummy])).fit()
            self.assertAlmostEqual(beta, fit.params[-1], places=10)


class HousingInputTests(unittest.TestCase):
    def test_real_time_income_uses_only_published_quarters(self):
        pipeline = load_module(PROJECTS[0], "pipeline")
        quarters = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(
            ["2020-03-31", "2020-06-30", "2020-09-30"]))
        with patch.object(pipeline, "load_income_quarterly", return_value=quarters):
            known = pipeline.real_time_income(lag_months=2)
        self.assertEqual(known.index[0], pd.Timestamp("2020-05-31"))
        self.assertEqual(known["2020-07-31"], 100.0)   # Q2 not yet published
        self.assertEqual(known["2020-08-31"], 110.0)
        self.assertEqual(known["2020-11-30"], 121.0)

    def test_stock_is_spliced_onto_the_2003_definition(self):
        pipeline = load_module(PROJECTS[0], "pipeline")
        stock = pd.DataFrame({"stock": [900.0, 1000.0, 950.0, 980.0],
                              "completions": [20.0, 30.0, 40.0, 30.0]},
                             index=[2001, 2002, 2003, 2004])
        spliced, factor = pipeline.splice_stock(stock)
        self.assertAlmostEqual(factor, (950.0 - 40.0) / 1000.0)
        self.assertAlmostEqual(spliced[2002], 950.0 - 40.0)
        self.assertEqual(list(spliced[[2003, 2004]]), [950.0, 980.0])

    def test_rolling_forecasts_reject_calendar_gaps(self):
        idx = pd.date_range("2000-01-31", periods=40, freq="ME")
        y = pd.Series(np.linspace(0, 1, 40), index=idx, name="y")
        x = pd.DataFrame({"x": np.linspace(1, 2, 40)}, index=idx)
        x.iloc[20] = np.nan
        for project in PROJECTS:
            with self.subTest(project=project):
                module = load_module(project, "forecast")
                with self.assertRaisesRegex(ValueError, "consecutive months"):
                    module.rolling_oos(y, x, horizons=(1,))


class OfficeEstimationTests(unittest.TestCase):
    def test_percent_text_cells_parse_as_rates(self):
        pipeline = load_module(PROJECTS[1], "pipeline")
        self.assertAlmostEqual(pipeline._coerce_num("9.5%"), 0.095)
        self.assertAlmostEqual(pipeline._coerce_num(" 12 % "), 0.12)
        self.assertAlmostEqual(pipeline._coerce_num("(58.1)"), 58.1)
        self.assertTrue(np.isnan(pipeline._coerce_num("-")))

    @staticmethod
    def annual(missing_year=None):
        rng = np.random.default_rng(21)
        years = range(1980, 2022)
        vac = pd.Series(0.09 + 0.03 * rng.standard_normal(len(years)), index=years)
        growth = 1.2 * (0.10 - vac.shift(1)) + 0.05 * rng.standard_normal(len(years))
        if missing_year is not None:
            vac[missing_year] = np.nan
        return pd.DataFrame({"dlr_A": growth, "vac_A": vac})

    def test_asymmetry_test_nests_the_symmetric_model(self):
        vacancy = load_module(PROJECTS[1], "vacancy")
        df = self.annual()
        result = vacancy.asymmetry(df, "A", n_boot=40)
        self.assertGreaterEqual(result["F_sym"], 0.0)
        d = pd.DataFrame({"y": df["dlr_A"], "v": df["vac_A"].shift(1)}).dropna()
        ols = sm.OLS(d["y"], sm.add_constant(d["v"])).fit()
        self.assertAlmostEqual(result["vstar_symmetric"],
                               ols.params["const"] / -ols.params["v"], places=10)
        # F against the symmetric fit with v* free (not v* fixed at the threshold)
        y, v = d["y"].to_numpy(), d["v"].to_numpy()
        ssr_threshold = []
        for vs in np.linspace(np.quantile(v, 0.15), np.quantile(v, 0.85), 61):
            gap, above = vs - v, (v > vs).astype(float)
            X = np.column_stack([gap * above, gap * (1 - above)])
            ssr_threshold.append(sm.OLS(y, X).fit().ssr)
        ssr_u = min(min(ssr_threshold), ols.ssr)
        expected = (ols.ssr - ssr_u) / (ssr_u / (len(y) - 3))
        self.assertAlmostEqual(result["F_sym"], expected, places=8)

    def test_time_varying_rate_keeps_gaps_and_finds_the_likelihood_maximum(self):
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        vacancy = load_module(PROJECTS[1], "vacancy")
        df = self.annual(missing_year=2000)
        info, path = vacancy.time_varying(df, "A", lam=1.2, n_boot=19)
        self.assertEqual(list(path.index), list(range(path.index[0], path.index[-1] + 1)))
        self.assertIn(2001, path.index)             # no v_{t-1}: a missing observation
        self.assertGreaterEqual(info["lr_constant"], 0.0)
        self.assertTrue(0.0 < info["p_constant"] <= 1.0)
        # the chosen fit beats the profile likelihood on a grid of level variances
        d = pd.DataFrame({"y": df["dlr_A"], "v": df["vac_A"].shift(1)})
        d = d.loc[path.index[0]:path.index[-1]]
        values = (d["y"] + 1.2 * d["v"]).to_numpy(float)
        model = UnobservedComponents(values, level="local level")
        for share in (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5):
            fit = model.fit_constrained({"sigma2.level": share * np.nanvar(values)},
                                        disp=False, maxiter=1000)
            self.assertGreaterEqual(info["loglike"], fit.llf - 1e-6)

    def test_office_bounds_test_uses_one_regressor_bounds(self):
        longrun = load_module(PROJECTS[1], "longrun")
        rng = np.random.default_rng(9)
        idx = pd.date_range("2000-01-31", periods=200, freq="ME")
        x = np.cumsum(rng.normal(size=200))
        panel = pd.DataFrame({"y": 0.5 * x + rng.normal(scale=0.5, size=200), "x": x},
                             index=idx)
        result = longrun.uecm_bounds(panel, "y", "x")
        self.assertEqual(result["k"], 1)
        self.assertAlmostEqual(result["F_cv95_lower"], 4.92, places=2)
        self.assertAlmostEqual(result["F_cv95_upper"], 5.72, places=2)

    def test_district_gaps_are_filled_only_when_short(self):
        breaks = load_module(PROJECTS[1], "breaks")
        idx = pd.date_range("2000-01-31", periods=24, freq="ME")
        s = pd.Series(np.arange(24.0), index=idx)
        s.iloc[3:5] = np.nan     # two months: filled
        s.iloc[12:16] = np.nan   # four months: left open
        filled = breaks.fill_short_gaps(s)
        self.assertEqual(filled.iloc[3], 3.0)
        self.assertTrue(filled.iloc[12:16].isna().all())
        run = breaks.longest_complete_run(pd.DataFrame({"a": filled, "b": 1.0}))
        self.assertEqual((run.index[0], len(run)), (idx[0], 12))

    def test_production_balancing_drops_the_late_entrant_months(self):
        breaks = load_module(PROJECTS[1], "breaks")
        rng = np.random.default_rng(6)
        idx = pd.date_range("2012-01-31", periods=156, freq="ME")
        levels = {"SheungWan": 0.1, "Central": 0.0, "CausewayBay": -0.1, "QuarryBay": -0.3,
                  "TST": -0.2, "MongKok": -0.4, "KwunTong": -1.1}
        dist = pd.DataFrame({f"ar_A_{d}": np.exp(5 + lvl + 0.01 * rng.normal(size=156)
                                                 + np.cumsum(0.002 * rng.normal(size=156)))
                             for d, lvl in levels.items()}, index=idx)
        dist.loc[:"2014-12-31", "ar_A_KwunTong"] = np.nan
        with tempfile.TemporaryDirectory() as out, patch.object(breaks, "OUT", out), \
                patch.object(breaks, "log", lambda message: None):
            result = breaks.convergence(dist)
        balanced = result["sigma_balanced"]
        self.assertEqual(balanced["start"], "2015-01-31")
        self.assertEqual(len(balanced["relatives"]), 6)
        self.assertLess(abs(balanced["post2019_mean"] - balanced["pre2019_mean"]), 0.02)
        self.assertGreater(result["sigma_all_observed"]["post2019_mean"]
                           - result["sigma_all_observed"]["pre2019_mean"], 0.05)
        self.assertIn("mean_change_hac_p", balanced)


if __name__ == "__main__":
    unittest.main()
