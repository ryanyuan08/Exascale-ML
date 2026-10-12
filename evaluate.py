"""Evaluate neural-network vs baseline Burgers' equation solver runs.

Loads every run directory in one experiment folder, compares each run to an
accurate Godunov reference that started from the SAME initial wave, and reports
per-run metrics:

  - ``runtime_ms``  : wall-clock solve time recorded in ``metadata.json``
  - ``err_l2_rel``  : relative L2 error vs the reference (main "solver error")
  - ``err_mae``     : mean absolute error vs the reference
  - ``err_max_abs`` : max absolute error vs the reference
  - ``err_final_l2_rel`` : relative L2 error at the final matched time
  - neural network improvement: for NN runs, the error reduction relative to
    an otherwise identical non-NN run at the same grid spacing, plus the
    runtime cost/speedup of using the NN.

Choosing the reference: each run is compared with the first reference in
``--gt`` (default: godunov_gt, then godunov_fine_grain) that has the same
initial wave and viscosity. So the dx sweep is scored against godunov_gt and the
NN test against godunov_fine_grain automatically. Runs with no matching
reference (e.g. the wiki cases) are kept in the CSV but marked
``ic_matches_reference = False`` and left out of the printed tables.

Solutions live on different grids (different ``dx``, ``dt``, and number of
points), so each run is interpolated in time and space onto the reference grid
before the error is computed.

Experiments: ``./burgers`` saves each experiment in its own folder,
``data/experiments/<name>/``. Results are written to
``<folder>/evaluation_results.csv``.

Usage (from the repo root, using the bundled deps from ``lib/python``):

    PYTHONPATH=lib/python python3 evaluate.py --list                       # show experiments
    PYTHONPATH=lib/python python3 evaluate.py --experiment nu_0p03_t_0p125 # evaluate one
    PYTHONPATH=lib/python python3 evaluate.py --all                        # evaluate every experiment
    PYTHONPATH=lib/python python3 evaluate.py --data-dir data              # any folder of runs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOLUTION_DATA_FILENAME = "solution.bin"
METADATA_FILENAME = "metadata.json"

DEFAULT_DATA_DIR = "data"
EXPERIMENTS_DIR = Path("data") / "experiments"
DEFAULT_GT_NAMES = ["godunov_gt", "godunov_fine_grain"]
DEFAULT_NUM_EVAL_TIMES = 2000


def load_run(run_dir, name):
    """Load a run directory into a dict of metadata + solution memmap."""
    bin_path = run_dir / SOLUTION_DATA_FILENAME
    meta_path = run_dir / METADATA_FILENAME
    if not bin_path.exists() or not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    solver = meta["solver"]
    n = int(solver["num_domain_points"])
    t = int(solver["time_steps"])
    dx = float(solver["spatial_step_size"])
    dt = float(solver["time_step_size"])
    domain_length = float(solver["domain_length"])
    nu = float(solver["kinematic_viscosity"])
    scheme_name = str(solver.get("scheme_name", "unknown"))
    use_nn = bool(solver.get("use_nn", False))
    runtime_ms = float(solver["runtime_ms"]) if "runtime_ms" in solver else np.nan
    nan_detected = bool(solver.get("nan_detected", False))

    time = dt * np.arange(t)
    x = dx * np.arange(n)
    u = np.memmap(bin_path, dtype=np.float64, mode="r", shape=(t, n))

    return {
        "name": name,
        "dir": run_dir,
        "scheme_name": scheme_name,
        "use_nn": use_nn,
        "dx": dx,
        "n": n,
        "dt": dt,
        "t": t,
        "domain_length": domain_length,
        "nu": nu,
        "final_time": float(time[-1]),
        "runtime_ms": runtime_ms,
        "nan_detected": nan_detected,
        "ic_key": ic_key(meta),
        "ic_matches_reference": None,
        "reference": None,
        "meta": meta,
        "x": x,
        "time": time,
        "u": u,
    }


def ic_key(meta):
    """Hashable fingerprint of the initial condition terms."""
    terms = meta.get("terms")
    if not isinstance(terms, list) or len(terms) == 0:
        return None
    bias = round(float(meta.get("bias", 0.0)), 8)
    term_tuple = tuple(
        sorted(
            (
                round(float(x["amplitude"]), 8),
                round(float(x["frequency"]), 8),
                round(float(x["phase_shift"]), 8),
            )
            for x in terms
        )
    )
    return (bias, term_tuple)


def interp_in_time(run, t_eval):
    """Vectorized linear interpolation of a run's solution in time.

    Returns an array of shape (len(t_eval), run.n).
    """
    t_pts = np.clip(t_eval, 0.0, run["final_time"])
    k = np.floor(t_pts / run["dt"]).astype(int)
    k = np.clip(k, 0, run["t"] - 2)
    frac = (t_pts / run["dt"]) - k
    u = run["u"]
    return u[k] * (1.0 - frac)[:, None] + u[k + 1] * frac[:, None]


def interp_in_space(run, x_target):
    """Vectorized periodic linear interpolation of a run's solution in space.

    The domain is periodic, so target points are wrapped into [0, n*dx).
    Returns an array of shape (..., len(x_target)).
    """
    period = run["n"] * run["dx"]
    dx = run["dx"]
    xt = np.mod(x_target, period)
    j = np.floor(xt / dx).astype(int)
    frac_x = (xt / dx) - j
    j = np.mod(j, run["n"])
    jp = np.mod(j + 1, run["n"])

    def apply(u_src):
        return u_src[..., j] * (1.0 - frac_x) + u_src[..., jp] * frac_x

    return apply


def compute_error_metrics(run, gt, t_eval):
    """Error of `run` vs `gt`, both evaluated on the gt grid and times."""
    u_g = gt["u"][t_eval]
    if run is gt:
        u_r = u_g
    else:
        u_r = interp_in_time(run, gt["time"][t_eval])
        u_r = interp_in_space(run, gt["x"])(u_r)

    diff = u_r - u_g
    denom = np.sqrt(np.nansum(u_g * u_g))
    err_l2_rel = np.sqrt(np.nansum(diff * diff)) / denom if denom > 0 else np.nan
    err_mae = np.nanmean(np.abs(diff))
    err_max_abs = np.nanmax(np.abs(diff))
    denom_final = np.sqrt(np.nansum(u_g[-1] * u_g[-1]))
    err_final_l2_rel = (
        np.sqrt(np.nansum(diff[-1] * diff[-1])) / denom_final
        if denom_final > 0
        else np.nan
    )
    return err_l2_rel, err_mae, err_max_abs, err_final_l2_rel


def find_baseline(run, runs):
    """Find the non-NN run with the same scheme, grid, and IC as `run`."""
    candidates = [
        r
        for r in runs
        if r["name"] != run["name"]
        and r["use_nn"] is False
        and r["scheme_name"] == run["scheme_name"]
        and abs(r["dx"] - run["dx"]) < 1e-12
        and abs(r["domain_length"] - run["domain_length"]) < 1e-12
        and abs(r["nu"] - run["nu"]) < 1e-12
        and r["t"] == run["t"]
        and run["ic_key"] is not None
        and r["ic_key"] == run["ic_key"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda r: abs(r["dx"] - run["dx"]))
    return candidates[0]


def improvement_pct(nn_val, base_val):
    if nn_val is None or base_val is None:
        return np.nan
    if not np.isfinite(nn_val) or not np.isfinite(base_val) or base_val == 0.0:
        return np.nan
    return (base_val - nn_val) / base_val * 100.0


def build_row(run, gt, baseline, metrics, experiment_name):
    err_l2_rel, err_mae, err_max_abs, err_final_l2_rel = metrics
    row = {
        "experiment": experiment_name,
        "run_name": run["name"],
        "scheme_name": run["scheme_name"],
        "use_nn": run["use_nn"],
        "kinematic_viscosity": run["nu"],
        "spatial_step_size": run["dx"],
        "num_domain_points": run["n"],
        "time_steps": run["t"],
        "time_step_size": run["dt"],
        "final_time": run["final_time"],
        "runtime_ms": run["runtime_ms"],
        "nan_detected": run["nan_detected"],
        "reference_run": gt["name"],
        "ic_matches_reference": run["ic_matches_reference"],
        "err_l2_rel": err_l2_rel,
        "err_mae": err_mae,
        "err_max_abs": err_max_abs,
        "err_final_l2_rel": err_final_l2_rel,
        "baseline_run": baseline["name"] if baseline is not None else "",
    }

    if run["use_nn"] and baseline is not None:
        base_err_l2, base_err_mae, base_err_max, base_err_final = compute_error_metrics(
            baseline, gt, gt["eval_idx"]
        )
        row["nn_improvement_l2_rel_pct"] = improvement_pct(err_l2_rel, base_err_l2)
        row["nn_improvement_mae_pct"] = improvement_pct(err_mae, base_err_mae)
        row["nn_improvement_final_pct"] = improvement_pct(err_final_l2_rel, base_err_final)
        if (
            np.isfinite(run["runtime_ms"])
            and np.isfinite(baseline["runtime_ms"])
            and baseline["runtime_ms"] > 0
        ):
            row["speedup_vs_baseline"] = baseline["runtime_ms"] / run["runtime_ms"]
            row["runtime_penalty_pct"] = (
                (run["runtime_ms"] - baseline["runtime_ms"]) / baseline["runtime_ms"] * 100.0
            )
        else:
            row["speedup_vs_baseline"] = np.nan
            row["runtime_penalty_pct"] = np.nan
    else:
        row["nn_improvement_l2_rel_pct"] = np.nan
        row["nn_improvement_mae_pct"] = np.nan
        row["nn_improvement_final_pct"] = np.nan
        row["speedup_vs_baseline"] = np.nan
        row["runtime_penalty_pct"] = np.nan

    return row


def make_formatter(spec):
    def fmt(value):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(f):
            return "n/a"
        return format(f, spec)

    return fmt


def print_run_table(df):
    matched = df[df["ic_matches_reference"]]
    skipped = df[~df["ic_matches_reference"]]
    cols = [
        "run_name",
        "reference_run",
        "scheme_name",
        "use_nn",
        "spatial_step_size",
        "num_domain_points",
        "time_steps",
        "runtime_ms",
        "nan_detected",
        "err_l2_rel",
        "err_mae",
        "err_max_abs",
        "err_final_l2_rel",
    ]
    table = matched[cols].sort_values(
        ["reference_run", "spatial_step_size", "use_nn", "run_name"],
        ascending=[True, True, False, True],
    )
    print("\nPer-run metrics (error vs the reference with the same starting wave):")
    print(
        table.to_string(
            index=False,
            na_rep="n/a",
            formatters={
                "spatial_step_size": make_formatter(".4f"),
                "runtime_ms": make_formatter(",.1f"),
                "err_l2_rel": make_formatter(".4e"),
                "err_mae": make_formatter(".4e"),
                "err_max_abs": make_formatter(".4e"),
                "err_final_l2_rel": make_formatter(".4e"),
            },
        )
    )


    if not skipped.empty:
        print(
            "\n(not shown: no reference with the same starting wave for "
            + ", ".join(skipped["run_name"])
            + "; their errors are in the CSV but are not meaningful)"
        )


def print_improvement_table(df):
    nn_df = df[df["use_nn"] & df["ic_matches_reference"]]
    if nn_df.empty:
        print("\nNo NN runs found.")
        return
    cols = [
        "run_name",
        "spatial_step_size",
        "baseline_run",
        "err_l2_rel",
        "nn_improvement_l2_rel_pct",
        "nn_improvement_mae_pct",
        "nn_improvement_final_pct",
        "runtime_ms",
        "speedup_vs_baseline",
        "runtime_penalty_pct",
    ]
    table = nn_df[cols].sort_values("spatial_step_size")
    print("\nNeural network vs non-NN baseline (same grid spacing / IC):")
    print(
        table.to_string(
            index=False,
            na_rep="n/a",
            formatters={
                "spatial_step_size": make_formatter(".4f"),
                "err_l2_rel": make_formatter(".4e"),
                "nn_improvement_l2_rel_pct": make_formatter("+.1f"),
                "nn_improvement_mae_pct": make_formatter("+.1f"),
                "nn_improvement_final_pct": make_formatter("+.1f"),
                "runtime_ms": make_formatter(",.1f"),
                "speedup_vs_baseline": make_formatter(".3f"),
                "runtime_penalty_pct": make_formatter("+,.1f"),
            },
        )
    )


def pick_reference(run, runs, gt_names):
    """First reference in gt_names with the same starting wave and viscosity as `run`."""
    for name in gt_names:
        ref = runs.get(name)
        if (
            ref is not None
            and run["ic_key"] is not None
            and ref["ic_key"] == run["ic_key"]
            and abs(ref["nu"] - run["nu"]) < 1e-12
        ):
            return ref, True
    # no match: fall back to the first available reference so the row still exists
    for name in gt_names:
        if name in runs:
            return runs[name], False
    return None, False


def evaluate_folder(data_dir, gt_names, output, num_eval_times):
    """Evaluate every run in data_dir. Returns the results DataFrame (or None)."""
    runs = {}
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        run = load_run(entry, entry.name)
        if run is not None:
            runs[entry.name] = run

    references = [name for name in gt_names if name in runs]
    if not references:
        print(
            f"error: none of the reference runs {gt_names} were found in {data_dir}. "
            f"Available runs: {sorted(runs)}",
            file=sys.stderr,
        )
        return None

    for name in references:
        ref = runs[name]
        num_eval = max(1, min(num_eval_times, ref["t"]))
        ref["eval_idx"] = np.linspace(0, ref["t"] - 1, num_eval).astype(int)

    experiment_name = data_dir.name
    rows = []
    for run in runs.values():
        ref, matches = pick_reference(run, runs, references)
        run["reference"] = ref
        run["ic_matches_reference"] = matches
        metrics = compute_error_metrics(run, ref, ref["eval_idx"])
        baseline = find_baseline(run, list(runs.values())) if run["use_nn"] else None
        rows.append(build_row(run, ref, baseline, metrics, experiment_name))

    df = pd.DataFrame(rows)
    output = Path(output) if output else data_dir / "evaluation_results.csv"
    df.to_csv(output, index=False)

    print("=" * 80)
    print(f"Experiment: {experiment_name}   ({data_dir})")
    config_path = data_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(
            f"viscosity = {config.get('kinematic_viscosity')}, "
            f"end time = {config.get('end_time')}, "
            f"seeds: nn = {config.get('nn_seed')}, dx = {config.get('dx_seed')}"
        )
    print(f"Wrote metrics CSV to: {output}")

    print_run_table(df)
    print_improvement_table(df)

    missing_baseline = df[(df["use_nn"]) & (df["baseline_run"] == "")]
    if not missing_baseline.empty:
        print(
            "\nnote: no non-NN baseline found for: "
            + ", ".join(missing_baseline["run_name"])
            + " (improvement columns left blank)."
        )
    return df


def list_experiments():
    if not EXPERIMENTS_DIR.is_dir():
        return []
    return sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--experiment",
                        help="experiment folder name inside data/experiments (e.g. nu_0p03_t_0p125)")
    target.add_argument("--all", action="store_true",
                        help="evaluate every experiment inside data/experiments")
    target.add_argument("--list", action="store_true",
                        help="list the experiments inside data/experiments and exit")
    target.add_argument("--data-dir",
                        help="any folder of run directories (default: data/)")
    parser.add_argument("--gt", nargs="+", default=DEFAULT_GT_NAMES,
                        help="reference run names, in order of preference (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: <folder>/evaluation_results.csv)")
    parser.add_argument("--num-eval-times", type=int, default=DEFAULT_NUM_EVAL_TIMES,
                        help="number of time samples used for error comparison "
                             "(default: %(default)s)")
    args = parser.parse_args()

    if args.list:
        experiments = list_experiments()
        if not experiments:
            print(f"No experiments found in {EXPERIMENTS_DIR}/")
        for path in experiments:
            done = "evaluated" if (path / "evaluation_results.csv").exists() else "not evaluated yet"
            print(f"  {path.name:30s} ({done})")
        return

    if args.all:
        folders = list_experiments()
        if not folders:
            print(f"error: no experiments found in {EXPERIMENTS_DIR}/", file=sys.stderr)
            sys.exit(1)
        if args.output:
            print("note: --output is ignored with --all (each experiment gets its own CSV)")
        for folder in folders:
            evaluate_folder(folder, args.gt, None, args.num_eval_times)
            print()
        return

    if args.experiment:
        data_dir = EXPERIMENTS_DIR / args.experiment
    else:
        data_dir = Path(args.data_dir or DEFAULT_DATA_DIR)

    if not data_dir.is_dir():
        print(f"error: folder not found: {data_dir}", file=sys.stderr)
        if args.experiment:
            print("Available experiments: " + ", ".join(p.name for p in list_experiments()),
                  file=sys.stderr)
        sys.exit(1)

    if evaluate_folder(data_dir, args.gt, args.output, args.num_eval_times) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()