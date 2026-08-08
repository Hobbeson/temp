#!/usr/bin/env python3
"""Compare FDS global HRR files from a baseline and a candidate calculation.

The script compares gas-phase heat release rate (HRR) and solid combustible
mass-loss rate (MLR) over one identical physical time interval.  It writes:

* comparison_summary.txt / .csv     -- numerical results for a test report
* comparison_samples.csv            -- uniformly spaced comparison points
* fds_hrr_comparison.png / .pdf     -- publication-style 2 x 2 figure

Example
-------
python compare_fds_hrr.py baseline_result candidate_result -o hrr_compare \
    --solid-mlr-columns MLR_POLYURETHANE

If there is exactly one *_hrr.csv beneath each result directory, file paths do
not need to be specified.  For multi-material cases, provide every combustible
solid material column explicitly, for example:

    --solid-mlr-columns MLR_WOOD,MLR_PMMA,MLR_CHAR

Requirements: Python 3.9+, numpy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RATE_SAMPLE_COUNT = 500
EXCLUDED_MLR_COLUMNS = {"MLR_AIR", "MLR_PRODUCTS", "MLR_TOTAL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HRR and solid-fuel mass-loss rate in two FDS *_hrr.csv outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline", type=Path, help="Baseline FDS result folder, or an *_hrr.csv file.")
    parser.add_argument("candidate", type=Path, help="Candidate FDS result folder, or an *_hrr.csv file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("fds_hrr_comparison"),
                        help="Directory for tables, text summary, and figures.")
    parser.add_argument("--baseline-hrr", type=Path, default=None,
                        help="Explicit baseline *_hrr.csv; overrides automatic discovery.")
    parser.add_argument("--candidate-hrr", type=Path, default=None,
                        help="Explicit candidate *_hrr.csv; overrides automatic discovery.")
    parser.add_argument("--solid-mlr-columns", default=None,
                        help=("Comma-separated solid combustible MLR columns to sum, e.g. "
                              "MLR_WOOD,MLR_PMMA. Strongly recommended for multi-material cases."))
    parser.add_argument("--time-span", choices=("common", "strict"), default="common",
                        help=("common: use the overlap of the two outputs; strict: require matching "
                              "start/end times (within --time-tolerance)."))
    parser.add_argument("--time-start", type=float, default=None,
                        help="Optional comparison start time in seconds; must lie in both outputs.")
    parser.add_argument("--time-end", type=float, default=None,
                        help="Optional comparison end time in seconds; must lie in both outputs.")
    parser.add_argument("--time-tolerance", type=float, default=1e-6,
                        help="Tolerance (s) used by strict time-span checking.")
    parser.add_argument("--zero-threshold", type=float, default=1e-12,
                        help=("Baseline magnitudes no larger than this are excluded from percentage-error "
                              "averages, because a relative error is undefined at zero."))
    parser.add_argument("--samples", type=int, default=DEFAULT_RATE_SAMPLE_COUNT,
                        help="Number of uniformly spaced points used to calculate error statistics.")
    return parser.parse_args()


def locate_hrr_file(source: Path, explicit: Path | None, label: str) -> Path:
    """Return a uniquely identified HRR CSV, otherwise give an actionable error."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} HRR file does not exist: {path}")
        return path

    source = source.expanduser().resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"{label} path does not exist: {source}")

    files = sorted(p for p in source.rglob("*_hrr.csv") if p.is_file())
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(f"No *_hrr.csv was found beneath {source}")
    file_list = "\n  ".join(str(p) for p in files)
    raise RuntimeError(
        f"Found {len(files)} possible {label} HRR files. Specify --{label}-hrr explicitly:\n  {file_list}"
    )


def read_fds_hrr(path: Path) -> pd.DataFrame:
    """Read the two-header-line FDS global HRR CSV and clean its time axis."""
    # Line 1 is units; line 2 contains names.  utf-8-sig also accepts ordinary UTF-8.
    try:
        df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, skiprows=1, encoding="latin1")
    df.columns = [str(c).strip() for c in df.columns]
    if "Time" not in df.columns:
        raise ValueError(f"{path} has no 'Time' column. Columns found: {list(df.columns)}")
    if "HRR" not in df.columns:
        raise ValueError(f"{path} has no 'HRR' column. Columns found: {list(df.columns)}")

    needed = ["Time", "HRR"] + [c for c in df.columns if c.upper().startswith("MLR_")]
    data = df.loc[:, list(dict.fromkeys(needed))].copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Time", "HRR"]).sort_values("Time")
    # Restart/appended files occasionally contain a duplicate output time. Keep the newest row.
    data = data.drop_duplicates(subset="Time", keep="last").reset_index(drop=True)
    if len(data) < 2:
        raise ValueError(f"{path} contains fewer than two valid HRR rows.")
    if not np.all(np.diff(data["Time"].to_numpy()) > 0):
        raise ValueError(f"{path} does not have a strictly increasing time column after cleaning.")
    return data


def choose_solid_mlr_columns(
    baseline: pd.DataFrame, candidate: pd.DataFrame, requested: str | None
) -> list[str]:
    """Select MLR columns without silently adding air or combustion products."""
    baseline_columns = set(baseline.columns)
    candidate_columns = set(candidate.columns)
    if requested:
        columns = [item.strip() for item in requested.split(",") if item.strip()]
        absent_a = [c for c in columns if c not in baseline_columns]
        absent_b = [c for c in columns if c not in candidate_columns]
        if absent_a or absent_b:
            raise ValueError(
                "Requested solid MLR columns must occur in both files. "
                f"Missing from baseline: {absent_a or 'none'}; missing from candidate: {absent_b or 'none'}."
            )
        return columns

    common = sorted(
        c for c in baseline_columns & candidate_columns
        if c.upper().startswith("MLR_") and c.upper() not in EXCLUDED_MLR_COLUMNS
    )
    # Older FDS files commonly provide one already aggregated solid-fuel output.
    if "MLR_FUEL" in common:
        return ["MLR_FUEL"]
    if len(common) == 1:
        return common
    if not common:
        raise ValueError(
            "No common solid-fuel MLR column was identified. Use --solid-mlr-columns "
            "to supply the material columns explicitly."
        )
    raise ValueError(
        "Several MLR columns could represent solid materials: " + ", ".join(common) + ". "
        "To avoid double counting, specify --solid-mlr-columns explicitly."
    )


def rate_from_columns(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    """Sum material loss rates; absent/blank output values are treated as zero only within a valid column."""
    return df.loc[:, list(columns)].fillna(0.0).sum(axis=1).to_numpy(dtype=float)


def interpolation(time: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    if target[0] < time[0] - 1e-9 or target[-1] > time[-1] + 1e-9:
        raise ValueError("Interpolation target lies outside an input time range.")
    return np.interp(target, time, values)


def cumulative_from_rate(time: np.ndarray, rate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Integrate a piecewise-linear rate exactly on a grid containing every target time.

    kW integrated over seconds gives kJ.  kg/s integrated over seconds gives kg.
    """
    grid = np.unique(np.concatenate(([target[0]], time[(time > target[0]) & (time < target[-1])], target)))
    values = interpolation(time, rate, grid)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(grid)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    return np.interp(target, grid, cumulative)


def mape(reference: np.ndarray, compared: np.ndarray, zero_threshold: float) -> tuple[float, int, np.ndarray]:
    valid = np.abs(reference) > zero_threshold
    if not np.any(valid):
        return float("nan"), 0, np.full(reference.shape, np.nan)
    point_error = np.full(reference.shape, np.nan)
    point_error[valid] = 100.0 * np.abs(compared[valid] - reference[valid]) / np.abs(reference[valid])
    return float(np.nanmean(point_error)), int(valid.sum()), point_error


def select_time_interval(a: np.ndarray, b: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    a0, a1 = float(a[0]), float(a[-1])
    b0, b1 = float(b[0]), float(b[-1])
    if args.time_span == "strict" and (abs(a0 - b0) > args.time_tolerance or abs(a1 - b1) > args.time_tolerance):
        raise ValueError(
            f"Strict mode requires identical output spans. Baseline: [{a0:g}, {a1:g}] s; "
            f"candidate: [{b0:g}, {b1:g}] s. Use --time-span common or set an explicit interval."
        )
    start = max(a0, b0) if args.time_start is None else args.time_start
    end = min(a1, b1) if args.time_end is None else args.time_end
    if start < max(a0, b0) - args.time_tolerance or end > min(a1, b1) + args.time_tolerance:
        raise ValueError("The requested time interval is not contained in both outputs.")
    if end <= start:
        raise ValueError(f"Invalid comparison interval: [{start:g}, {end:g}] s.")
    return float(start), float(end)


def publication_figure(
    output: Path,
    plot_time: np.ndarray,
    hrr_a: np.ndarray,
    hrr_b: np.ndarray,
    mlr_a: np.ndarray,
    mlr_b: np.ndarray,
    m_a: np.ndarray,
    m_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = ("#1f77b4", "#d62728")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], hrr_a, hrr_b, "Heat release rate (kW)", "(a) Gas-phase heat release rate"),
        (axes[1], mlr_a, mlr_b, "Mass loss rate (kg s$^{-1}$)", "(b) Solid combustible mass-loss rate"),
        (axes[2], m_a, m_b, "Cumulative mass loss (kg)", "(c) Cumulative solid mass loss"),
    ]
    for axis, y_a, y_b, ylabel, title in panels:
        axis.plot(plot_time, y_a, color=colors[0], lw=1.8, label=label_a)
        axis.plot(plot_time, y_b, color=colors[1], lw=1.5, ls="--", label=label_b)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(True, alpha=0.25, lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes:
        axis.set_xlabel("Time (s)")
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.samples < 2:
        print("ERROR: --samples must be at least 2.", file=sys.stderr)
        return 2
    try:
        file_a = locate_hrr_file(args.baseline, args.baseline_hrr, "baseline")
        file_b = locate_hrr_file(args.candidate, args.candidate_hrr, "candidate")
        df_a = read_fds_hrr(file_a)
        df_b = read_fds_hrr(file_b)
        solid_columns = choose_solid_mlr_columns(df_a, df_b, args.solid_mlr_columns)

        time_a, time_b = df_a["Time"].to_numpy(), df_b["Time"].to_numpy()
        hrr_a, hrr_b = df_a["HRR"].to_numpy(), df_b["HRR"].to_numpy()
        mlr_a, mlr_b = rate_from_columns(df_a, solid_columns), rate_from_columns(df_b, solid_columns)
        t_start, t_end = select_time_interval(time_a, time_b, args)
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    sample_count = args.samples
    sample_time = np.linspace(t_start, t_end, sample_count + 1)[1:]
    hrr_a_s = interpolation(time_a, hrr_a, sample_time)
    hrr_b_s = interpolation(time_b, hrr_b, sample_time)
    mlr_a_s = interpolation(time_a, mlr_a, sample_time)
    mlr_b_s = interpolation(time_b, mlr_b, sample_time)
    m_a_s = cumulative_from_rate(time_a, mlr_a, np.r_[t_start, sample_time])[1:]
    m_b_s = cumulative_from_rate(time_b, mlr_b, np.r_[t_start, sample_time])[1:]

    hrr_mape, hrr_n, hrr_error = mape(hrr_a_s, hrr_b_s, args.zero_threshold)
    mlr_mape, mlr_n, mlr_error = mape(mlr_a_s, mlr_b_s, args.zero_threshold)
    mass_mape, mass_n, mass_error = mape(m_a_s, m_b_s, args.zero_threshold)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.DataFrame({
        "Time_s": sample_time,
        "HRR_baseline_kW": hrr_a_s,
        "HRR_candidate_kW": hrr_b_s,
        "HRR_absolute_percentage_error_pct": hrr_error,
        "Solid_MLR_baseline_kg_s": mlr_a_s,
        "Solid_MLR_candidate_kg_s": mlr_b_s,
        "Solid_MLR_absolute_percentage_error_pct": mlr_error,
        "Cumulative_mass_baseline_kg": m_a_s,
        "Cumulative_mass_candidate_kg": m_b_s,
        "Cumulative_mass_absolute_percentage_error_pct": mass_error,
    })
    samples.to_csv(output_dir / "comparison_samples.csv", index=False, float_format="%.10g")

    summary = pd.DataFrame([
        ("Gas-phase heat release rate", "kW", hrr_mape, hrr_n),
        ("Solid combustible mass-loss rate", "kg/s", mlr_mape, mlr_n),
        ("Cumulative solid mass loss", "kg", mass_mape, mass_n),
    ], columns=["Quantity", "Unit", "Mean_absolute_percentage_error_pct", "Valid_sample_count"])
    summary.to_csv(output_dir / "comparison_summary.csv", index=False, float_format="%.10g")

    label_a, label_b = "Baseline FDS", "Candidate FDS"
    plot_time = np.linspace(t_start, t_end, 1001)
    hrr_a_p, hrr_b_p = interpolation(time_a, hrr_a, plot_time), interpolation(time_b, hrr_b, plot_time)
    mlr_a_p, mlr_b_p = interpolation(time_a, mlr_a, plot_time), interpolation(time_b, mlr_b, plot_time)
    m_a_p, m_b_p = cumulative_from_rate(time_a, mlr_a, plot_time), cumulative_from_rate(time_b, mlr_b, plot_time)
    publication_figure(output_dir / "fds_hrr_comparison", plot_time, hrr_a_p, hrr_b_p, mlr_a_p, mlr_b_p,
                       m_a_p, m_b_p, label_a, label_b)

    lines = [
        "FDS HRR/MLR comparison summary",
        "=" * 72,
        f"Baseline file : {file_a}",
        f"Candidate file: {file_b}",
        f"Comparison time interval: {t_start:.10g} to {t_end:.10g} s (duration {t_end - t_start:.10g} s)",
        f"Solid combustible MLR columns summed: {', '.join(solid_columns)}",
        f"Uniform samples: {sample_count} points; endpoint included; start point excluded.",
        f"Percentage-error reference: baseline; |baseline| <= {args.zero_threshold:g} is excluded.",
        "",
        f"Gas-phase HRR MAPE                 = {hrr_mape:.6g} %  ({hrr_n}/{sample_count} valid points)",
        f"Solid combustible MLR MAPE         = {mlr_mape:.6g} %  ({mlr_n}/{sample_count} valid points)",
        f"Cumulative solid mass loss MAPE    = {mass_mape:.6g} %  ({mass_n}/{sample_count} valid points)",
        "",
        "Notes:",
        "- HRR is the FDS gas-phase heat release rate in the HRR column.",
        "- Solid mass-loss rate is the sum of the listed MLR material columns; MLR_AIR and",
        "  MLR_PRODUCTS are deliberately excluded.",
        "- Cumulative mass is the time integral of solid MLR (kg/s*s = kg), reset to zero",
        "  at the comparison start.",
    ]
    summary_text = "\n".join(lines) + "\n"
    (output_dir / "comparison_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Tables and figures written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
