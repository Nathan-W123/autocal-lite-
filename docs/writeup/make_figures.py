#!/usr/bin/env python3
"""Figures for the AutoCal-lite write-up, drawn from the committed campaign results.

Reads results/mc/{convergence,drift_tracking_plot,knock,knock_no_detector} and writes
fig_headline.png, fig_convergence.png and fig_knock.png next to this script.

  python3 docs/writeup/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MC = ROOT / "results" / "mc"
OUT = Path(__file__).resolve().parent

# Reference categorical palette (light), validated all-pairs for these three slots.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLUMN_IN = 3.35  # width of one column of the two-column page [in]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 6.8,
    "savefig.dpi": 300,
})


def load(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def envelope(ax, t, p50, p95, mx, color, label, scale=1.0):
    """Median line, p50-p95 band and a thin max line for one series."""
    ax.fill_between(t, [scale * v for v in p50], [scale * v for v in p95], color=color,
                    alpha=0.18, linewidth=0)
    ax.plot(t, [scale * v for v in mx], color=color, linewidth=0.6, linestyle=":")
    ax.plot(t, [scale * v for v in p50], color=color, linewidth=1.2, label=label)


def log_decades(ax, decades):
    """Label every decade and draw minor ticks, so a log axis can be read at print size."""
    ax.set_yticks(decades)
    ax.yaxis.set_minor_locator(LogLocator(subs=range(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def limit(ax, y, text, x_text, va="bottom"):
    ax.axhline(y, color=MUTED, linewidth=0.7, linestyle="--")
    ax.text(x_text, y, f" {text}", color=MUTED, fontsize=6.6, va=va, ha="left")


def headline() -> None:
    conv = load(MC / "convergence" / "envelope.csv")
    drift = load(MC / "drift_tracking_plot.csv")
    fig, (a, b) = plt.subplots(2, 1, figsize=(COLUMN_IN, 2.4))

    # (a) Convergence from 5 deg / 5 cm, rotation error over the first 60 s.
    envelope(a, conv["time_s"], conv["rot_err_p50_deg"], conv["rot_err_p95_deg"],
             conv["rot_err_max_deg"], BLUE, "rotation error")
    a.set_yscale("log")
    a.set_xlim(0, 60)
    a.set_ylim(0.005, 12)
    log_decades(a, [0.01, 0.1, 1, 10])
    limit(a, 0.5, "0.5° criterion", 38)
    a.set_ylabel("rotation error [°]")
    a.set_xlabel("time [s]")
    a.set_title("(a) Convergence from a 5° / 5 cm error, 50 runs", loc="left")

    # (b) Drift: truth rotates 0.1 deg/min from t = 60 s.
    t_min = [(t - 60.0) / 60.0 for t in drift["time_s"]]
    keep = [i for i, t in enumerate(t_min) if t >= 0.0]
    pick = lambda key: [drift[key][i] for i in keep]  # noqa: E731
    tt = [t_min[i] for i in keep]
    envelope(b, tt, pick("online_err_p50_deg"), pick("online_err_p95_deg"),
             pick("online_err_max_deg"), BLUE, "online (random-walk extrinsic)")
    b.fill_between(tt, pick("frozen_err_p50_deg"), pick("frozen_err_p95_deg"), color=ORANGE,
                   alpha=0.18, linewidth=0)
    b.plot(tt, pick("frozen_err_p50_deg"), color=ORANGE, linewidth=1.2,
           label="mount assumed fixed (no random walk)")
    limit(b, 0.3, "0.3° criterion", 0.2)
    b.set_xlim(0, 10)
    b.set_ylim(0, 0.6)
    b.set_ylabel("rotation error [°]")
    b.set_xlabel("minutes after drift onset at t = 60 s (0.1°/min)")
    b.set_title("(b) Tracking a drifting mount, 50 runs", loc="left")
    b.text(2.4, 0.40, "mount assumed fixed", color=ORANGE, fontsize=6.8)
    b.text(6.2, 0.085, "online", color=BLUE, fontsize=6.8)

    fig.tight_layout(h_pad=1.0)
    fig.savefig(OUT / "fig_headline.png")
    plt.close(fig)


def convergence_detail() -> None:
    conv = load(MC / "convergence" / "envelope.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COLUMN_IN, 1.9))
    envelope(a, conv["time_s"], conv["rot_err_p50_deg"], conv["rot_err_p95_deg"],
             conv["rot_err_max_deg"], BLUE, "rotation")
    envelope(b, conv["time_s"], conv["trans_err_p50_m"], conv["trans_err_p95_m"],
             conv["trans_err_max_m"], BLUE, "lever arm", scale=100.0)
    for ax, lim, txt, unit in ((a, 0.5, "0.5° criterion", "rotation error [°]"),
                               (b, 1.0, "1 cm criterion", "lever-arm error [cm]")):
        ax.set_yscale("log")
        ax.set_xlim(0, 60)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(unit)
        limit(ax, lim, txt, 40)
    a.set_ylim(0.005, 12)
    b.set_ylim(0.03, 12)
    log_decades(a, [0.01, 0.1, 1, 10])
    log_decades(b, [0.1, 1, 10])
    a.set_title("(a) Rotation", loc="left")
    b.set_title("(b) Lever arm", loc="left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_convergence.png")
    plt.close(fig)


def knock() -> None:
    on = load(MC / "knock" / "envelope.csv")
    off = load(MC / "knock_no_detector" / "envelope.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COLUMN_IN, 1.9))
    window = lambda d, key, s=1.0: [s * v for t, v in zip(d["time_s"], d[key])  # noqa: E731
                                     if 85.0 <= t <= 130.0]
    t_on = window(on, "time_s")
    t_off = window(off, "time_s")
    envelope(a, t_on, window(on, "rot_err_p50_deg"), window(on, "rot_err_p95_deg"),
             window(on, "rot_err_max_deg"), BLUE, "with change detector")
    a.plot(t_off, window(off, "rot_err_p50_deg"), color=ORANGE, linewidth=1.2,
           label="detector off (random walk only)")
    envelope(b, t_on, window(on, "trans_err_p50_m", 100.0), window(on, "trans_err_p95_m", 100.0),
             window(on, "trans_err_max_m", 100.0), BLUE, "with change detector")
    b.plot(t_off, window(off, "trans_err_p50_m", 100.0), color=ORANGE, linewidth=1.2,
           label="detector off (random walk only)")
    for ax, lim, txt, unit in ((a, 0.5, "0.5°", "rotation error [°]"),
                               (b, 1.0, "1 cm", "lever-arm error [cm]")):
        ax.axvline(90.0, color=MUTED, linewidth=0.6)
        ax.set_xlim(85, 130)
        ax.set_xlabel("time [s] (knock at 90 s)")
        ax.set_ylabel(unit)
        limit(ax, lim, txt, 120)
    a.set_ylim(0, 2.2)
    b.set_ylim(0, 4.5)
    a.set_title("(a) Rotation", loc="left")
    b.set_title("(b) Lever arm", loc="left")
    a.legend(loc="upper right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_knock.png")
    plt.close(fig)


if __name__ == "__main__":
    headline()
    convergence_detail()
    knock()
    print(f"figures written to {OUT}")
