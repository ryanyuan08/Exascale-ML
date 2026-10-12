"""Combine every evaluated experiment into one table and a few comparison graphs.

Run evaluate.py on each experiment first, then (from the repo root):

    PYTHONPATH=lib/python python3 scripts/compare_experiments.py
    PYTHONPATH=lib/python python3 scripts/compare_experiments.py --experiments nu_0p8_t_0p125 nu_0p03_t_0p125

Reads   data/experiments/<name>/evaluation_results.csv
Writes  results/all_experiments.csv          every row from every experiment
        results/nn_improvement_table.csv     NN improvement (%) by grid spacing x experiment
        results/error_vs_dx.png              error vs grid spacing, with and without the NN
        results/nn_improvement_vs_dx.png     NN improvement vs grid spacing
        results/runtime_vs_error.png         accuracy vs cost for every dx sweep run

Only the dx sweep (runs scored against godunov_gt with the same starting wave) is
used for the graphs, because it is the only part with matched NN / no-NN pairs at
several grid spacings.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "data" / "experiments"
RESULTS_DIR = ROOT / "results"

# Categorical colors, assigned to experiments in a fixed order (never cycled).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3df"


def experiment_label(name, config):
    """Readable legend label, e.g. 'nu = 0.03, t = 0.125'."""
    if config and "kinematic_viscosity" in config:
        label = f"nu = {config['kinematic_viscosity']:g}"
        if "end_time" in config:
            label += f", t = {config['end_time']:g}"
        return label
    return name


def load_experiments(names):
    frames, labels = [], {}
    folders = (
        [EXPERIMENTS_DIR / n for n in names]
        if names
        else sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
    )
    for folder in folders:
        csv_path = folder / "evaluation_results.csv"
        if not csv_path.exists():
            print(f"skipping {folder.name}: not evaluated yet "
                  f"(run: evaluate.py --experiment {folder.name})")
            continue
        config = None
        config_path = folder / "experiment_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        df = pd.read_csv(csv_path)
        df["experiment"] = folder.name
        if config:
            df["end_time"] = config.get("end_time", np.nan)
        frames.append(df)
        labels[folder.name] = experiment_label(folder.name, config)
    if not frames:
        return None, labels
    return pd.concat(frames, ignore_index=True), labels


def dx_sweep_rows(df):
    """NN / no-NN pairs from the dx sweep, one row per (experiment, dx)."""
    sweep = df[
        (df["reference_run"] == "godunov_gt")
        & (df["ic_matches_reference"])
        & (df["run_name"].str.match(r"^(nn|ftcs)_dx_"))
    ].copy()
    sweep["kind"] = np.where(sweep["use_nn"], "nn", "ftcs")
    return sweep


def experiment_order(df, labels):
    """Sort experiments by viscosity (then name) so colors stay stable between runs."""
    nu_by_exp = df.groupby("experiment")["kinematic_viscosity"].first()
    return sorted(labels, key=lambda e: (-nu_by_exp.get(e, 0.0), e))


def dx_ticks(ax, dx_values):
    """Label the log x axis with the actual grid spacings (0.01, 0.02, ...)."""
    ticks = sorted(set(round(v, 6) for v in dx_values))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_error_vs_dx(sweep, order, labels, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, exp in enumerate(order):
        color = COLORS[i % len(COLORS)]
        rows = sweep[sweep["experiment"] == exp].sort_values("spatial_step_size")
        for kind, style, marker, suffix in [("ftcs", "--", "o", "no NN"), ("nn", "-", "s", "with NN")]:
            part = rows[rows["kind"] == kind]
            part = part[np.isfinite(part["err_l2_rel"]) & (part["err_l2_rel"] < 1e3)]
            if part.empty:
                continue
            ax.plot(part["spatial_step_size"], part["err_l2_rel"], linestyle=style, marker=marker,
                    markersize=7, linewidth=2, color=color, label=f"{labels[exp]} ({suffix})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    dx_ticks(ax, sweep["spatial_step_size"])
    style_axes(ax, "Solver error vs grid spacing", "grid spacing dx (coarser →)",
               "relative L2 error vs godunov_gt")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_improvement_vs_dx(pairs, order, labels, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, exp in enumerate(order):
        rows = pairs[pairs["experiment"] == exp].sort_values("spatial_step_size")
        rows = rows[np.isfinite(rows["nn_improvement_l2_rel_pct"])]
        if rows.empty:
            continue
        ax.plot(rows["spatial_step_size"], rows["nn_improvement_l2_rel_pct"], marker="o",
                markersize=7, linewidth=2, color=COLORS[i % len(COLORS)], label=labels[exp])
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.text(0.01, 0.02, "below 0 = NN made it worse", transform=ax.transAxes,
            color=TEXT_SECONDARY, fontsize=9)
    ax.set_xscale("log")
    dx_ticks(ax, pairs["spatial_step_size"])
    style_axes(ax, "NN improvement vs grid spacing", "grid spacing dx (coarser →)",
               "error reduction from the NN (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_runtime_vs_error(sweep, order, labels, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, exp in enumerate(order):
        color = COLORS[i % len(COLORS)]
        rows = sweep[sweep["experiment"] == exp]
        for kind, marker, suffix in [("ftcs", "o", "no NN"), ("nn", "s", "with NN")]:
            part = rows[(rows["kind"] == kind) & np.isfinite(rows["runtime_ms"])
                        & (rows["runtime_ms"] > 0) & (rows["err_l2_rel"] < 1e3)]
            if part.empty:
                continue
            ax.scatter(part["runtime_ms"], part["err_l2_rel"], marker=marker, s=60,
                       color=color if kind == "nn" else "white", edgecolors=color, linewidths=2,
                       label=f"{labels[exp]} ({suffix})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, "Accuracy vs cost (lower-left is better)", "runtime (ms)",
               "relative L2 error vs godunov_gt")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluated experiments.")
    parser.add_argument("--experiments", nargs="+",
                        help="experiment names to include (default: every evaluated experiment)")
    args = parser.parse_args()

    if not EXPERIMENTS_DIR.is_dir():
        print(f"error: {EXPERIMENTS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    df, labels = load_experiments(args.experiments)
    if df is None:
        print("error: no evaluated experiments found", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "all_experiments.csv", index=False)

    sweep = dx_sweep_rows(df)
    if sweep.empty:
        print("No dx sweep runs found; wrote results/all_experiments.csv only.")
        return
    order = experiment_order(df, labels)
    pairs = sweep[sweep["kind"] == "nn"]

    # Table: NN improvement (%) by grid spacing, one column per experiment
    table = pairs.pivot_table(index="spatial_step_size", columns="experiment",
                              values="nn_improvement_l2_rel_pct")
    table = table[[e for e in order if e in table.columns]]
    table.columns = [labels[c] for c in table.columns]
    table.index.name = "dx"
    table.to_csv(RESULTS_DIR / "nn_improvement_table.csv")

    blew_up = sweep[(sweep["nan_detected"]) | ~np.isfinite(sweep["err_l2_rel"]) | (sweep["err_l2_rel"] >= 1e3)]

    print("NN improvement in relative L2 error (%), dx sweep. Positive = NN helped.\n")
    print(table.to_string(float_format=lambda v: f"{v:+.1f}"))
    if not blew_up.empty:
        print("\nBlew up (left off the graphs): "
              + ", ".join(f"{r.experiment}/{r.run_name}" for r in blew_up.itertuples()))

    plot_error_vs_dx(sweep, order, labels, RESULTS_DIR / "error_vs_dx.png")
    plot_improvement_vs_dx(pairs, order, labels, RESULTS_DIR / "nn_improvement_vs_dx.png")
    plot_runtime_vs_error(sweep, order, labels, RESULTS_DIR / "runtime_vs_error.png")

    print(f"\nSaved to {RESULTS_DIR.relative_to(ROOT)}/: all_experiments.csv, nn_improvement_table.csv, "
          "error_vs_dx.png, nn_improvement_vs_dx.png, runtime_vs_error.png")


if __name__ == "__main__":
    main()