"""
make_figures.py — Result figures. Each function is idempotent and writes
PNG (300 dpi) + PDF to figures/. Run: python3 make_figures.py [fig1 fig2 ...]
(no args = all available given current outputs).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pipeline import build_panel
from policy import MEASURES

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FIGD = os.path.join(ROOT, "figures")
OUTD = os.path.join(ROOT, "output")
os.makedirs(FIGD, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

MEASURE_LABELS = {"SSD": "SSD\nNov 10", "BSD": "BSD\nOct 12", "DSD": "DSD\nFeb 13",
                  "NRSD": "NRSD\nNov 16", "EASE": "All removed\nFeb 24"}
# SSD, BSD and DSD fall within 28 months: stagger their labels on three rows
LABEL_OFFSET_PT = {"BSD": -24, "DSD": -46}
CRISES = [("1997-10-31", "Asian financial crisis"), ("2003-04-30", "SARS trough"),
          ("2008-09-30", "GFC"), ("2019-06-30", "Social unrest"),
          ("2020-01-31", "COVID-19"), ("2022-03-31", "Fed tightening")]


def _save(fig, name):
    fig.savefig(os.path.join(FIGD, f"{name}.png"))
    fig.savefig(os.path.join(FIGD, f"{name}.pdf"))
    plt.close(fig)
    print(f"figures/{name}.png")


def _policy_vlines(ax, ymax_frac=0.97, label=True):
    for name, date in MEASURES.items():
        d = pd.Timestamp(date)
        ax.axvline(d, color="firebrick" if name != "EASE" else "seagreen",
                   lw=0.9, ls="--", alpha=0.8)
        if label:
            ax.annotate(MEASURE_LABELS[name], xy=(d, ax.get_ylim()[1]),
                        xytext=(2, LABEL_OFFSET_PT.get(name, -2)),
                        textcoords="offset points",
                        fontsize=6.5, va="top",
                        color="firebrick" if name != "EASE" else "seagreen")


def fig1_indices(panel):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(panel.index, panel["px_ALL"], lw=1.2, color="navy",
            label="Price index (all classes)")
    ax.plot(panel.index, panel["rt_ALL"], lw=1.2, color="darkorange",
            label="Rental index (all classes)")
    ax.set_ylabel("Index (1999 = 100)")
    ax.set_ylim(0, panel["px_ALL"].max() * 1.12)
    _policy_vlines(ax)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("RVD private domestic price and rental indices, 1993–2026")
    _save(fig, "fig1_indices")


def fig2_real_and_ratio(panel):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    ax = axes[0]
    ax.plot(panel.index, panel["lp_real"], lw=1.1, color="navy",
            label="log real price")
    ax.plot(panel.index, panel["lr_real"], lw=1.1, color="darkorange",
            label="log real rent")
    ax.set_ylabel("log(index / CPI)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Real (CPI-deflated) price and rent")
    ax2 = axes[1]
    ax2.plot(panel.index, panel["lpr"], lw=1.1, color="seagreen")
    ax2.axhline(panel["lpr"].mean(), color="gray", lw=0.8, ls=":")
    ax2.set_ylabel("log(P/R)")
    ax2.set_title("Log price-rent ratio (sample mean dotted)")
    for a in axes:
        _policy_vlines(a, label=False)
    _save(fig, "fig2_real_ratio")


def fig3_bsadf(panel):
    series = [("lp_nom", "log nominal price"), ("lp_real", "log real price"),
              ("lpr", "log price-rent ratio")]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    for ax, (name, title) in zip(axes, series):
        df = pd.read_csv(os.path.join(OUTD, f"bsadf_{name}.csv"),
                         index_col=0, parse_dates=True)
        res = json.load(open(os.path.join(OUTD, "bubbles.json")))[name]
        # shade the episodes dated against the Monte-Carlo 95% sequence
        for i, ep in enumerate(res["episodes"]):
            ax.axvspan(pd.Timestamp(ep["start"]), pd.Timestamp(ep["end"]),
                       color="lightsteelblue", alpha=0.65, lw=0,
                       label="dated episode (MC)" if i == 0 else None)
        ax.plot(df.index, df["bsadf"], lw=0.9, color="navy", label="BSADF$_t$")
        ax.plot(df.index, df["cv95"], lw=0.8, color="firebrick", ls="--",
                label="95% CV (Monte Carlo)")
        ax.plot(df.index, df["cv95_wb"], lw=0.8, color="darkorange", ls=":",
                label="95% pointwise CV (wild bootstrap)")
        ax.set_ylim(-4, 3.2)
        ax.set_title(f"{title}:  GSADF = {res['gsadf']:.2f} "
                     f"(MC 95% CV {res['mc_cv']['0.95']:.2f}, "
                     f"wild-bootstrap p = {res['wb_pvalue']:.2f})", fontsize=9)
        if ax is axes[0]:
            ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    fig.suptitle("Backward sup-ADF date-stamping, k = 1, w₀ = 40 months",
                 y=0.995, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig3_bsadf")


def fig4_equilibrium():
    eq = pd.read_csv(os.path.join(OUTD, "equilibrium_path_S3H.csv"),
                     index_col=0, parse_dates=True)
    s3h = json.load(open(os.path.join(OUTD, "longrun.json")))["specs"]["S3H"]["full"]
    beta = s3h["dols_beta"]["real_hibor"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(eq.index, eq["lpr"], lw=1.2, color="navy",
            label="log price-rent ratio (actual)")
    ax.plot(eq.index, eq["fitted"], lw=1.2, color="firebrick", ls="--",
            label=f"long-run fitted: ${beta:.3f} \\times$ real HIBOR (DOLS)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Price-rent ratio vs its rate-implied equilibrium (S3-HIBOR)")
    ax2 = axes[1]
    ax2.fill_between(eq.index, eq["ect"], 0, color="seagreen", alpha=0.5)
    ax2.axhline(0, color="black", lw=0.7)
    half_life = (f"{s3h['half_life_m']:.1f} months" if s3h["half_life_m"] is not None
                 else "none (λ outside (−1, 0))")
    ax2.set_title(f"Equilibrium error (log deviation); ECM half-life {half_life}")
    for a in axes:
        _policy_vlines(a, label=False)
    fig.tight_layout()
    _save(fig, "fig4_equilibrium")


def fig5_lp_irfs():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    panels = [("lp_tightening_price", "Tightening → cumulative log price"),
              ("lp_tightening_volume", "Tightening → cumulative log volume"),
              ("lp_ease_price", "2024 removal → cumulative log price"),
              ("lp_ease_volume", "2024 removal → cumulative log volume")]
    for ax, (name, title) in zip(axes.ravel(), panels):
        p = os.path.join(OUTD, f"{name}.csv")
        if not os.path.exists(p):
            ax.set_visible(False)
            continue
        irf = pd.read_csv(p, index_col=0)
        ax.fill_between(irf.index, irf["lo95"], irf["hi95"], color="lightsteelblue",
                        alpha=0.6, lw=0)
        ax.fill_between(irf.index, irf["lo90"], irf["hi90"], color="steelblue",
                        alpha=0.4, lw=0)
        ax.plot(irf.index, irf["beta"], lw=1.4, color="navy")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("months after adoption")
    fig.suptitle("Jordà local projections: dynamic response to policy events\n"
                 "(90% and 95% bands from placebo event dates)", y=0.995, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig5_lp_irfs")


def fig6_forecast():
    summ = pd.read_csv(os.path.join(OUTD, "oos_summary.csv"))
    oos12 = pd.read_csv(os.path.join(OUTD, "oos_h12.csv"), index_col=0,
                        parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    piv = summ.pivot(index="model", columns="horizon", values="rel_RW")
    piv = piv.loc[["RWD", "AR", "ECM", "GBM"]]
    piv.plot(kind="bar", ax=ax, color=["#c6dbef", "#6baed6", "#2171b5"], rot=0,
             legend=True)
    ax.axhline(1.0, color="firebrick", lw=1.0, ls="--")
    ax.set_ylabel("RMSE relative to random walk")
    ax.set_title("Out-of-sample RMSE ratios")
    ax.legend(title="horizon (m)", frameon=False)
    ax2 = axes[1]
    ax2.plot(oos12.index, oos12["actual"], color="black", lw=1.3, label="actual")
    for m, c in [("RW", "gray"), ("ECM", "firebrick"), ("GBM", "seagreen")]:
        ax2.plot(oos12.index, oos12[m], lw=0.9, alpha=0.85, color=c, label=m)
    ax2.set_title("12-month-ahead forecasts vs actual")
    ax2.set_xlabel("Forecast origin")
    ax2.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, "fig6_forecast")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or None
    panel = build_panel()
    jobs = {"fig1": lambda: fig1_indices(panel),
            "fig2": lambda: fig2_real_and_ratio(panel),
            "fig3": lambda: fig3_bsadf(panel),
            "fig4": fig4_equilibrium,
            "fig5": fig5_lp_irfs,
            "fig6": fig6_forecast}
    for name, job in jobs.items():
        if which and name not in which:
            continue
        try:
            job()
        except FileNotFoundError as e:
            print(f"skip {name}: missing input ({e.filename})")
