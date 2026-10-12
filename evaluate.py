"""Evaluate neural-network vs baseline Burgers' equation solver runs.

Loads every run directory under a data directory (default: ``data/``), compares
each run's solution to the ``godunov_gt`` ground truth, and reports per-run
metrics:

  - ``runtime_ms``  : wall-clock solve time recorded in ``metadata.json``
  - ``err_l2_rel``  : relative L2 error vs godunov_gt (main "solver error")
  - ``err_mae``     : mean absolute error vs godunov_gt
  - ``err_max_abs`` : max absolute error vs godunov_gt
  - ``err_final_l2_rel`` : relative L2 error at the final matched time
  - neural network improvement: for NN runs, the error reduction relative to
    an otherwise identical non-NN run at the same grid spacing, plus the
    runtime cost/speedup of using the NN.

Solutions live on different grids (different ``dx``, ``dt``, and number of
points), so each run is interpolated in time and space onto the ground truth
grid before the error is computed.

Writes a CSV (default: ``data/evaluation_results.csv``) and prints comparison
tables to stdout.

Usage (uses the bundled deps from ``lib/python``):

    PYTHONPATH=lib/python python3 evaluate.py
    PYTHONPATH=lib/python python3 evaluate.py --data-dir data --gt godunov_gt
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
DEFAULT_GT_NAME = "godunov_gt"
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
        "ic_matches_gt": None,
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


def build_row(run, gt, baseline, metrics):
    err_l2_rel, err_mae, err_max_abs, err_final_l2_rel = metrics
    row = {
        "run_name": run["name"],
        "scheme_name": run["scheme_name"],
        "use_nn": run["use_nn"],
        "spatial_step_size": run["dx"],
        "num_domain_points": run["n"],
        "time_steps": run["t"],
        "time_step_size": run["dt"],
        "final_time": run["final_time"],
        "runtime_ms": run["runtime_ms"],
        "nan_detected": run["nan_detected"],
        "ic_matches_gt": run["ic_matches_gt"],
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


def print_run_table(df, gt_name):
    cols = [
        "run_name",
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
    table = df[cols].sort_values(
        ["spatial_step_size", "use_nn", "run_name"], ascending=[True, False, True]
    )
    print("\nPer-run metrics (error vs %s ground truth):" % gt_name)
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


def print_improvement_table(df):
    nn_df = df[df["use_nn"]]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="directory containing run folders (default: %(default)s)")
    parser.add_argument("--gt", default=DEFAULT_GT_NAME,
                        help="name of the ground truth run (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: <data-dir>/evaluation_results.csv)")
    parser.add_argument("--num-eval-times", type=int, default=DEFAULT_NUM_EVAL_TIMES,
                        help="number of time samples used for error comparison "
                             "(default: %(default)s)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    runs = {}
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        run = load_run(entry, entry.name)
        if run is not None:
            runs[entry.name] = run

    if args.gt not in runs:
        print(
            f"error: ground truth run '{args.gt}' not found in {data_dir}. "
            f"Available runs: {sorted(runs)}",
            file=sys.stderr,
        )
        sys.exit(1)

    gt = runs[args.gt]
    num_eval = max(1, min(args.num_eval_times, gt["t"]))
    gt["eval_idx"] = np.linspace(0, gt["t"] - 1, num_eval).astype(int)

    for run in runs.values():
        run["ic_matches_gt"] = (
            run["ic_key"] is not None and run["ic_key"] == gt["ic_key"]
        )

    rows = []
    for run in runs.values():
        metrics = compute_error_metrics(run, gt, gt["eval_idx"])
        baseline = find_baseline(run, list(runs.values())) if run["use_nn"] else None
        rows.append(build_row(run, gt, baseline, metrics))

    df = pd.DataFrame(rows)

    output = Path(args.output) if args.output else data_dir / "evaluation_results.csv"
    df.to_csv(output, index=False)
    print(f"Wrote metrics CSV to: {output}")

    if df.empty:
        print("No runs evaluated.", file=sys.stderr)
        sys.exit(0)

    print_run_table(df, args.gt)
    print_improvement_table(df)

    missing_baseline = df[(df["use_nn"]) & (df["baseline_run"] == "")]
    if not missing_baseline.empty:
        print(
            "\nnote: no non-NN baseline found for: "
            + ", ".join(missing_baseline["run_name"])
            + " (improvement columns left blank)."
        )


if __name__ == "__main__":
    main()