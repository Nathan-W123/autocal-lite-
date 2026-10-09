#!/usr/bin/env python3
"""AutoCal-lite evaluation: run the Monte-Carlo campaigns and report against the criteria.

Prints
  1. the convergence table   (criterion 1: 5 deg / 5 cm prior -> < 0.5 deg and < 1 cm within
                              60 s in >= 95 % of 50 runs),
  2. the drift-tracking table (criterion 2: truth drifts at 0.1 deg/min after convergence;
                              tracking error < 0.3 deg), with the calibrate-once baseline,
  3. the supplementary scenarios (time offset, observability ablation, outliers, knock),
  4. the drift-tracking plot data (also written to <output>/drift_tracking_plot.csv),
and, when Matplotlib is installed, renders results/figures/{convergence,drift_tracking}.png.

Only the Python standard library is required; Matplotlib is optional.

Usage:
  python3 python/scripts/eval.py                 # run everything (50 trials) and report
  python3 python/scripts/eval.py --skip-run      # report on existing results/mc
  python3 python/scripts/eval.py --trials 10     # quicker, for CI
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ["convergence", "drift", "drift_frozen"]
EXTRA = ["time_offset", "time_offset_ignored", "yaw_only", "outliers", "knock",
         "knock_no_detector"]


def run_campaigns(binary: Path, scenarios: list[str], trials: int, output: Path, threads: int) -> None:
    cmd = [str(binary), "--scenario", ",".join(scenarios), "--trials", str(trials),
           "--output", str(output)]
    if threads > 0:
        cmd += ["--threads", str(threads)]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_summary(output: Path, scenario: str) -> dict | None:
    path = output / scenario / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        summary = json.load(f)
    summary["stats"] = {s["name"]: s for s in summary["statistics"]}
    return summary


def load_csv(path: Path) -> list[dict[str, float]]:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def fmt(stat: dict, keys=("median", "p95", "max"), scale=1.0, digits=3) -> str:
    return " / ".join(f"{scale * stat[k]:.{digits}f}" for k in keys)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report(output: Path) -> int:
    failures = 0
    conv = load_summary(output, "convergence")
    drift = load_summary(output, "drift")
    frozen = load_summary(output, "drift_frozen")

    # ---------------------------------------------------------------- criterion 1
    print("\n## 1. Convergence from a 5 deg / 5 cm miscalibration (60 s of data)\n")
    if conv is None:
        print("(no results/mc/convergence - run without --skip-run)")
        failures += 1
    else:
        s = conv["stats"]
        rate = conv["convergence_rate"]
        ok = rate >= 0.95
        failures += not ok
        print("| metric | value |")
        print("|---|---|")
        print(f"| runs | {conv['trials']} (master seed {conv['master_seed']}) |")
        print(f"| prior error | {conv['prior_rotation_error_deg']:.1f} deg, "
              f"{100 * conv['prior_translation_error_m']:.1f} cm (random direction per run) |")
        print(f"| converged (< {conv['rotation_threshold_deg']} deg and "
              f"< {100 * conv['translation_threshold_m']:.0f} cm at {conv['convergence_horizon_s']:.0f} s) "
              f"| **{conv['converged']}/{conv['trials']} = {100 * rate:.1f} %** "
              f"(target >= 95 %: {verdict(ok)}) |")
        print(f"| rotation error at 60 s, p50 / p95 / max | {fmt(s['horizon_rotation_error'])} deg |")
        print(f"| translation error at 60 s, p50 / p95 / max | "
              f"{fmt(s['horizon_translation_error'], digits=2)} cm |")
        print(f"| time to converge (stays converged), p50 / p95 / max | "
              f"{fmt(s['convergence_time'], digits=1)} s |")
        print(f"| gyro / accel bias error at 60 s, p50 | "
              f"{s['final_gyro_bias_error']['median']:.2e} rad/s / "
              f"{s['final_accel_bias_error']['median']:.2e} m/s^2 |")
        print(f"| mean extrinsic NEES (6 = consistent, < 6 = conservative) | "
              f"{s['mean_nees']['mean']:.2f} |")
        print(f"| mean vision NIS per dof (1 = consistent) | {s['mean_nis_per_dof']['mean']:.3f} |")
        print(f"| landmarks fused per frame, mean | {s['mean_landmarks_used']['mean']:.0f} |")
        print(f"| change-detector false alarms | {s['change_events']['mean'] * s['change_events']['count']:.0f} |")
        print(f"| wall clock per 60 s run, mean | {s['trial_wall_clock']['mean']:.2f} s |")

    # ---------------------------------------------------------------- criterion 2
    print("\n## 2. Drift tracking: truth rotates at 0.1 deg/min for 10 min after convergence\n")
    if drift is None:
        print("(no results/mc/drift)")
        failures += 1
    else:
        print("| estimator | runs with max error < 0.3 deg | max error p50 / p95 / max [deg] "
              "| RMS error p50 / p95 [deg] | truth moved [deg] |")
        print("|---|---|---|---|---|")
        for name, summ, scen in (("online (random-walk extrinsic + change detector)", drift, "drift"),
                                 ("static-extrinsic baseline (no random walk)", frozen, "drift_frozen")):
            if summ is None:
                continue
            s = summ["stats"]
            moved = [r["truth_drift_total_deg"] for r in load_csv(output / scen / "trials.csv")]
            print(f"| {name} | {summ['tracked']}/{summ['trials']} "
                  f"({100 * summ['tracking_rate']:.0f} %) | {fmt(s['tracking_max_error'])} | "
                  f"{fmt(s['tracking_rms_error'], keys=('median', 'p95'))} | "
                  f"{sum(moved) / max(len(moved), 1):.2f} |")
        ev = drift["stats"]["change_events"]
        print(f"\nChange-detector false alarms during the slow drift: {ev['mean'] * ev['count']:.0f} "
              f"in {drift['trials']} runs.")
        # Every run must have completed, been scored over the drift window and stayed under
        # 0.3 deg (failed runs count against it rather than being dropped).
        ok = drift["tracked"] == drift["trials"]
        failures += not ok
        print(f"\nCriterion 2 (tracking error < 0.3 deg throughout the drift, every run): "
              f"{drift['tracked']}/{drift['trials']} -> {verdict(ok)}")

    # ---------------------------------------------------------------- extras
    rows = [(n, load_summary(output, n)) for n in EXTRA]
    rows = [(n, r) for n, r in rows if r is not None]
    if rows:
        print("\n## 3. Supplementary scenarios\n")
        print("| scenario | converged @ 60 s | rot err @ 60 s p50 / p95 [deg] "
              "| trans err @ 60 s p50 / p95 [cm] | extra |")
        print("|---|---|---|---|---|")
        for name, r in rows:
            s = r["stats"]
            extra = ""
            if name == "time_offset":
                extra = f"t_d error p50 / p95: {fmt(s['final_time_offset_error'], keys=('median', 'p95'), digits=2)} ms"
            elif name == "time_offset_ignored":
                extra = "2.5 ms camera lag, t_d estimation switched off"
            elif name.startswith("knock"):
                rec = s["recovery_time"]
                recovered = f"{rec['count']}/{r['trials']} back under 0.5 deg & 1 cm"
                if rec["count"] > 0:
                    recovered += f" after {rec['median']:.1f} s (p50) / {rec['p95']:.1f} s (p95)"
                det = s.get("detection_latency", {"count": 0})
                detected = (f"detected after {det['median']:.2f} s (p50) / {det['max']:.2f} s (max); "
                            if det["count"] > 0 else "")
                extra = (f"1.5 deg / 2 cm knock at 90 s: {detected}{recovered}; error at 180 s p50 "
                         f"{s['final_rotation_error']['median']:.3f} deg / "
                         f"{s['final_translation_error']['median']:.2f} cm; detector fired "
                         f"{s['change_events']['mean']:.1f}x per run")
            elif name == "yaw_only":
                extra = ("roll/pitch excitation removed: lever arm along the yaw axis unobservable")
            elif name == "outliers":
                extra = "5 % wrong correspondences"
            print(f"| {name} | {r['converged']}/{r['trials']} | "
                  f"{fmt(s['horizon_rotation_error'], keys=('median', 'p95'))} | "
                  f"{fmt(s['horizon_translation_error'], keys=('median', 'p95'), digits=2)} | {extra} |")

    # ---------------------------------------------------------------- plot data
    env_path = output / "drift" / "envelope.csv"
    env = load_csv(env_path) if drift is not None and env_path.exists() else []
    if env:
        frozen_env = {}
        if frozen is not None and (output / "drift_frozen" / "envelope.csv").exists():
            frozen_env = {round(r["time_s"], 3): r for r in load_csv(output / "drift_frozen" / "envelope.csv")}
        rows_out = []
        for r in env:
            t = round(r["time_s"], 3)
            fz = frozen_env.get(t)
            rows_out.append({
                "time_s": t,
                "truth_drift_p50_deg": r["truth_drift_p50_deg"],
                "estimate_drift_p50_deg": r["estimate_drift_p50_deg"],
                "online_err_p50_deg": r["rot_err_p50_deg"],
                "online_err_p95_deg": r["rot_err_p95_deg"],
                "online_err_max_deg": r["rot_err_max_deg"],
                "frozen_err_p50_deg": fz["rot_err_p50_deg"] if fz else float("nan"),
                "frozen_err_p95_deg": fz["rot_err_p95_deg"] if fz else float("nan"),
            })
        path = output / "drift_tracking_plot.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n## 4. Drift-tracking plot data ({path.relative_to(ROOT) if path.is_relative_to(ROOT) else path})\n")
        print("| t [s] | truth drift [deg] | estimate [deg] | online err p50 / p95 / max [deg] "
              "| frozen err p50 [deg] |")
        print("|---|---|---|---|---|")
        for r in rows_out:
            if r["time_s"] < 50 or int(round(r["time_s"])) % 60 != 0:
                continue
            print(f"| {r['time_s']:.0f} | {r['truth_drift_p50_deg']:.3f} | "
                  f"{r['estimate_drift_p50_deg']:.3f} | {r['online_err_p50_deg']:.3f} / "
                  f"{r['online_err_p95_deg']:.3f} / {r['online_err_max_deg']:.3f} | "
                  f"{r['frozen_err_p50_deg']:.3f} |")
    return failures


def figures(output: Path, figdir: Path) -> None:
    conv_summary = load_summary(output, "convergence")
    n_conv = conv_summary["trials"] if conv_summary else 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping figures; pip install matplotlib)")
        return
    figdir.mkdir(parents=True, exist_ok=True)
    # Validated colour-blind-safe pair reused from Aether6's style.
    blue, orange, grey = "#0072B2", "#D55E00", "#7f7f7f"

    conv_path = output / "convergence" / "envelope.csv"
    if conv_path.exists():
        env = load_csv(conv_path)
        t = [r["time_s"] for r in env]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for ax, key, scale, unit, thr, title in (
                (axes[0], "rot_err", 1.0, "deg", 0.5, "Rotation error"),
                (axes[1], "trans_err", 100.0, "cm", 1.0, "Translation error")):
            suffix = "_deg" if key == "rot_err" else "_m"
            ax.fill_between(t, [scale * r[f"{key}_p50{suffix}"] for r in env],
                            [scale * r[f"{key}_max{suffix}"] for r in env], color=blue, alpha=0.15,
                            label="p50 - max")
            ax.plot(t, [scale * r[f"{key}_p50{suffix}"] for r in env], color=blue, lw=1.8, label="median")
            ax.plot(t, [scale * r[f"{key}_p95{suffix}"] for r in env], color=blue, lw=1.0, ls="--",
                    label="95th percentile")
            ax.axhline(thr, color=orange, lw=1.2, label=f"target {thr:g} {unit}")
            ax.set_yscale("log")
            ax.set_xlabel("time [s]")
            ax.set_ylabel(f"{title.lower()} [{unit}]")
            ax.set_title(f"{title} over {n_conv} runs (5 deg / 5 cm start)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.savefig(figdir / "convergence.png", dpi=150)
        plt.close(fig)

    plot_path = output / "drift_tracking_plot.csv"
    drift_summary = load_summary(output, "drift")
    if plot_path.exists():
        # Plot from the drift onset: the 5 deg initial convergence is the other figure's job
        # and would flatten the 0.05 deg tracking error on a linear axis.
        start = drift_summary.get("tracking_start_s", 0.0) if drift_summary else 0.0
        rows = [r for r in load_csv(plot_path) if r["time_s"] >= start]
        t = [r["time_s"] / 60.0 for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        ax = axes[0]
        ax.plot(t, [r["truth_drift_p50_deg"] for r in rows], color=grey, lw=2.5, label="truth")
        ax.plot(t, [r["estimate_drift_p50_deg"] for r in rows], color=blue, lw=1.4, ls="--",
                label="online estimate")
        ax.set_xlabel("time [min]")
        ax.set_ylabel("rotation from the initial mount [deg]")
        ax.set_title("Mount drifting at 0.1 deg/min from t = 1 min (median)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.fill_between(t, [r["online_err_p50_deg"] for r in rows], [r["online_err_max_deg"] for r in rows],
                        color=blue, alpha=0.15)
        ax.plot(t, [r["online_err_p50_deg"] for r in rows], color=blue, lw=1.8, label="online (median, band to max)")
        ax.plot(t, [r["frozen_err_p50_deg"] for r in rows], color=grey, lw=1.4, label="random walk + detector off (median)")
        ax.axhline(0.3, color=orange, lw=1.2, label="target 0.3 deg")
        ax.set_ylim(0.0, 0.65)
        ax.set_xlabel("time [min]")
        ax.set_ylabel("rotation error [deg]")
        ax.set_title(f"Tracking error during the drift, {drift_summary['trials'] if drift_summary else 0} runs")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(figdir / "drift_tracking.png", dpi=150)
        plt.close(fig)
    print(f"\nfigures written to {figdir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "mc")
    ap.add_argument("--binary", type=Path, default=ROOT / "build" / "bin" / "autocal_mc")
    ap.add_argument("--skip-run", action="store_true", help="only report on existing results")
    ap.add_argument("--core-only", action="store_true", help="skip the supplementary scenarios")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="where to write the PNGs (default: results/figures for results/mc, "
                         "else <output>/figures)")
    args = ap.parse_args()
    # Campaigns run with cwd=ROOT, so resolve paths against the caller's directory first.
    args.output = args.output.resolve()
    args.binary = args.binary.resolve()

    if not args.skip_run:
        if not args.binary.exists():
            print(f"{args.binary} not found - build first (make build)", file=sys.stderr)
            return 2
        scenarios = CORE if args.core_only else CORE + EXTRA
        run_campaigns(args.binary, scenarios, args.trials, args.output, args.threads)

    failures = report(args.output)
    if not args.no_figures:
        figdir = args.figures_dir
        if figdir is None:
            default_out = (ROOT / "results" / "mc").resolve()
            figdir = ROOT / "results" / "figures" if args.output.resolve() == default_out \
                else args.output / "figures"
        figures(args.output, figdir)
    print(f"\nOverall: {'ALL CRITERIA MET' if failures == 0 else f'{failures} criterion/criteria not met'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
