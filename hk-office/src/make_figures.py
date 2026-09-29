"""
make_figures.py — result figures (PNG, 150 dpi) from processed data and
estimation outputs.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PRO = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
C = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "TOT": "#444444",
     "ALL": "#444444"}


def load():
    panel = pd.read_csv(os.path.join(PRO, "monthly_panel.csv"),
                        index_col=0, parse_dates=True)
    ann = pd.read_csv(os.path.join(PRO, "annual_stockflow.csv"), index_col=0)
    dist = pd.read_csv(os.path.join(PRO, "district_monthly.csv"),
                       index_col=0, parse_dates=True)
    return panel, ann, dist


def fig1_overview(panel, ann):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    for g in ("A", "B", "C"):
        ax.plot(panel.index, panel[f"rt_{g}"], lw=1.2, color=C[g],
                label=f"Grade {g} rent")
        ax.plot(panel.index, panel[f"px_{g}"], lw=0.9, color=C[g], ls="--",
                alpha=0.7, label=f"Grade {g} price")
    ax.set_yscale("log")
    ax.set_title("(a) Office rent (—) and price (--) indices, 1999=100")
    ax.legend(ncol=2, frameon=False)

    ax = axes[0, 1]
    for g in ("A", "B", "C", "TOT"):
        s = ann[f"vacrate_{g}"].dropna()
        ax.plot(s.index, 100 * s, lw=1.4 if g == "TOT" else 1.0, color=C[g],
                label="Total" if g == "TOT" else f"Grade {g}")
    ax.set_title("(b) Year-end vacancy rate (%)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    comp = ann["comp_TOT"].dropna() / 1000.0
    tk = ann["takeup_TOT"].dropna() / 1000.0
    ax.bar(comp.index, comp, color="#bbbbbb", label="Completions")
    ax.plot(tk.index, tk, color="#d62728", lw=1.3, label="Take-up")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(c) Completions and net take-up ('000 m², all offices)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for g in ("A", "B", "C"):
        s = (100 * panel[f"yld_{g}_mean"]).dropna()
        ax.plot(s.index, s, lw=1.1, color=C[g], label=f"Grade {g}")
    rh = panel["real_hibor"].dropna()
    ax.plot(rh.index, rh, lw=0.8, color="#9467bd", alpha=0.8,
            label="real 1m HIBOR")
    ax.set_title("(d) Office yields from RVD levels vs real HIBOR (% p.a.)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_market_overview.png"))
    plt.close(fig)


def fig2_caprate(panel):
    lr = json.load(open(os.path.join(OUT, "longrun.json")))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(panel.index, panel["lpr_A"], color=C["A"], lw=1.2)
    ax.set_title("(a) log price–rent ratio, Grade A")
    ax2 = ax.twinx()
    ax2.plot(panel.index, panel["real_hibor"], color="#9467bd", lw=0.8, alpha=0.7)
    ax2.set_ylabel("real HIBOR (%)", color="#9467bd")
    ax2.grid(False); ax2.spines["right"].set_visible(True)

    ax = axes[1]
    d = panel[["lpr_A", "real_hibor"]].dropna()
    ax.scatter(d["real_hibor"], d["lpr_A"], s=6, alpha=0.45, color=C["A"])
    b = lr["dols_lpr_A_real_hibor"]["beta"]
    xs = np.linspace(d["real_hibor"].min(), d["real_hibor"].max(), 50)
    a0 = d["lpr_A"].mean() - b * d["real_hibor"].mean()
    ax.plot(xs, a0 + b * xs, color="k", lw=1.2,
            label=f"DOLS slope = {b:.3f}")
    ax.set_xlabel("real 1m HIBOR (%)"); ax.set_ylabel("log P/R (Grade A)")
    ax.set_title("(b) Levels relation (1996–2026)")
    ax.legend(frameon=False)

    ax = axes[2]
    cs = lr["campbell_shiller"]
    shares = [cs["share_rent_growth"], cs["share_riskfree"],
              cs["share_risk_premium"]]
    cis = [cs["ci_rent"], cs["ci_riskfree"], None]
    names = ["expected\nrent growth", "real prime\nrate news", "premium over\nprime rate"]
    bars = ax.bar(names, shares, color=["#2ca02c", "#9467bd", "#d62728"],
                  alpha=0.85)
    for i, ci in enumerate(cis):
        if ci:
            ax.errorbar(i, shares[i], yerr=[[shares[i] - ci[0]], [ci[1] - shares[i]]],
                        fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(c) Variance shares of log P/R\n(Campbell–Shiller VAR, 95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_caprate_channel.png"))
    plt.close(fig)


def fig3_vacancy(ann):
    vac = json.load(open(os.path.join(OUT, "vacancy.json")))
    vp = pd.read_csv(os.path.join(OUT, "vstar_path.csv"), index_col=0).sort_index()
    q = pd.read_csv(os.path.join(PRO, "quarterly_indices.csv"), index_col=0)
    q.index = pd.PeriodIndex(q.index, freq="Q")
    q4 = q[q.index.quarter == 4].copy(); q4.index = q4.index.year

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, g in zip(axes[:2], ("A", "TOT")):
        gq = "ALL" if g == "TOT" else g
        rr = np.log(q4[f"rt_{gq}"] / q4["cpi"]).diff()
        v = ann[f"vacrate_{g}"].shift(1)
        d = pd.concat([rr.rename("dlr"), v.rename("vlag")], axis=1).dropna()
        years = d.index
        sc = ax.scatter(100 * d["vlag"], 100 * d["dlr"], s=18,
                        c=years, cmap="viridis")
        b = [x for x in vac["baseline"] if x["grade"] == g][0]
        xs = np.linspace(d["vlag"].min(), d["vlag"].max(), 40)
        ax.plot(100 * xs, 100 * (b["alpha"] - b["lambda"] * xs), "k-", lw=1.2)
        ax.axvline(100 * b["vstar"], color="#d62728", ls=":", lw=1.2,
                   label=f"v* = {100*b['vstar']:.1f}%")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("vacancy rate t−1 (%)")
        ax.set_ylabel("Δ log real rent (%)")
        ttl = "Grade A" if g == "A" else "All offices"
        ax.set_title(f"({'ab'[g=='TOT']}) Rent adjustment, {ttl}")
        ax.legend(frameon=False)
        plt.colorbar(sc, ax=ax, shrink=0.8)

    ax = axes[2]
    ax.plot(vp.index, 100 * vp["vac_TOT"], color="#444", lw=1.3,
            label="vacancy (total)")
    ax.plot(vp.index, 100 * vp["vstar_TOT"], color="#d62728", lw=1.2, ls="--",
            label="time-varying v* (total)")
    if "vstar_C" in vp.columns:
        ax.plot(vp.index, 100 * vp["vstar_C"], color=C["C"], lw=1.0, ls=":",
                label="time-varying v* (Grade C)")
    ax.set_title("(c) Vacancy vs smoothed natural rate")
    ax.set_ylabel("%"); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_vacancy_rent.png"))
    plt.close(fig)


def fig4_simulation(ann):
    sim = pd.read_csv(os.path.join(OUT, "simulation_paths.csv"))
    sf = json.load(open(os.path.join(OUT, "stockflow.json")))
    vstar = sf["sim_params"]["vstar"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hist = ann["vacrate_TOT"].dropna()
    ax.plot(hist.index, 100 * hist, color="#444", lw=1.5, label="history (total)")
    colors = {"halt": "#2ca02c", "base": "#1f77b4", "heavy": "#d62728"}
    styles = {"weak": ":", "base": "-", "strong": "--"}
    for scen, g in sim.groupby("scenario"):
        gdp = scen.split("|")[0].split("_")[1]
        pipe = scen.split("|")[1].split("_")[1]
        ax.plot(g["year"], 100 * g["vacancy"], color=colors[pipe],
                ls=styles[gdp], lw=1.3 if gdp == "base" else 0.9,
                label=f"GDP {gdp}, pipeline {pipe}" if gdp == "base" else None)
    ax.axhline(100 * vstar, color="k", ls="-.", lw=1,
               label=f"natural rate v* = {100*vstar:.1f}%")
    ax.set_xlim(2005, 2035.5)
    ax.set_ylabel("vacancy rate (%)")
    ax.set_title("Vacancy-overhang absorption: four-quadrant simulation 2026–2035\n"
                 "(line style: GDP weak ⋯ / base — / strong − −;  colour: pipeline)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_simulation.png"))
    plt.close(fig)


def fig5_regimes(panel, dist):
    br = json.load(open(os.path.join(OUT, "breaks.json")))
    pr = pd.read_csv(os.path.join(OUT, "ms_probs.csv"), index_col=0,
                     parse_dates=True)
    sig = pd.read_csv(os.path.join(OUT, "sigma_path.csv"), index_col=0,
                      parse_dates=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    ax = axes[0]
    y = 100 * panel["lr_real_A"].diff()
    ax.plot(y.index, y, color=C["A"], lw=0.7)
    # shade the high-volatility regime (the one with the larger sigma)
    mskey = "p_regime0" if br["markov"]["sigma_regime0"] > br["markov"]["sigma_regime1"] \
        else None
    p_volatile = pr["p_regime0"] if mskey else 1.0 - pr["p_regime0"]
    ax.fill_between(pr.index, -12, 12, where=p_volatile > 0.5,
                    color="#d62728", alpha=0.15, label="volatile regime (P>0.5)")
    for bd in br["bp_rent_growth"]["break_dates"]:
        ax.axvline(pd.Timestamp(bd), color="k", ls=":", lw=1)
    ax.set_ylim(-12, 12)
    ax.set_title("(a) Grade A real rent growth (%/m):\nMS volatile regime + Bai–Perron breaks")
    ax.legend(frameon=False)

    ax = axes[1]
    # Kwun Tong enters in 2013-02, so only the balanced lines keep the
    # district set fixed; the all-observed line jumps when it enters.
    ax.plot(sig.index, sig["sigma_all_observed"], color="#bbb", lw=1.0,
            label="all observed")
    ax.plot(sig.index, sig["sigma_balanced_ex_kwuntong"], color="#1f77b4",
            lw=1.0, label="balanced, excl. Kwun Tong")
    ax.plot(sig.index, sig["sigma_balanced"], color="#444", lw=1.2,
            label="balanced, all districts")
    ax.axvline(pd.Timestamp("2019-01-01"), color="#d62728", ls=":", lw=1.2,
               label="2019 split")
    ax.set_title("(b) σ-dispersion of district log rents\nrelative to Central (Grade A)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    prem = (np.log(dist["ar_A_Central"]) - np.log(dist["ar_A_KwunTong"])).dropna()
    ax.plot(prem.index, prem, color="#1f77b4", lw=1.0)
    for bd in br["bp_central_premium"]["break_dates"]:
        ax.axvline(pd.Timestamp(bd), color="k", ls=":", lw=1)
    for m, (a, b) in zip(br["bp_central_premium"]["segment_means"],
                         [(prem.index[0], br["bp_central_premium"]["break_dates"][0]),
                          (br["bp_central_premium"]["break_dates"][0],
                           br["bp_central_premium"]["break_dates"][1]),
                          (br["bp_central_premium"]["break_dates"][1], prem.index[-1])]):
        ax.hlines(m, pd.Timestamp(a), pd.Timestamp(b), color="#d62728", lw=1.4)
    ax.set_title("(c) Central / Kwun Tong log rent premium\nwith Bai–Perron segment means")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_regimes_districts.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    panel, ann, dist = load()
    fig1_overview(panel, ann)
    fig2_caprate(panel)
    fig3_vacancy(ann)
    fig4_simulation(ann)
    fig5_regimes(panel, dist)
    print("figures written:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
