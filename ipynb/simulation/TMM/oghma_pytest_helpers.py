"""Shared helpers for Oghma alignment pytest modules in this directory."""

from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

WL_UM_XLABEL = "λ (μm)"


# Default R/T gates vs reflect/transmit.csv (optical filter / tight alignment).
RT_R_RMSE = 0.005
RT_R_MAX = 0.012
RT_R_CORR = 0.999
RT_T_RMSE = 0.001
RT_T_MAX = 0.002
# Emission equivalent-plane-wave path uses per-λ float32 solves; allow tiny drift vs passive spectrum.
CROSS_PATH_RTOL = 1e-6


@dataclass
class PlotSeries:
    x: np.ndarray
    ours: np.ndarray
    baseline: np.ndarray
    ylabel: str
    title: str
    xlabel: str = "V (V)"
    log_domain: bool = False
    filename: str | None = None


@dataclass
class CompareCaseResult:
    name: str
    passed: bool
    skipped: bool = False
    stats: dict[str, float] | None = None
    message: str = ""
    plots: list[PlotSeries] = field(default_factory=list)


def log10_series(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(values, 1e-30))


def wl_grid_um(start_nm: float, stop_nm: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    wl_nm = np.linspace(start_nm, stop_nm, n)
    return wl_nm * 1e-3, wl_nm


def log_before_visualize(description: str, **details: Any) -> None:
    from compare_data import py_pipe

    parts = [description]
    for key, value in details.items():
        parts.append(f"{key}={value}")
    py_pipe.log(" | ".join(parts))
    print(f"[viz] {description}", flush=True)
    for key, value in details.items():
        print(f"      {key}: {value}", flush=True)


def plot_compare_1d(
    x: np.ndarray,
    ours: np.ndarray,
    baseline: np.ndarray,
    *,
    ylabel: str,
    title: str,
    filename: str,
    out_dir: Path,
    plot_1d_compare,
    log_domain: bool = False,
    baseline_name: str = "Oghma FDTD",
    tmm_label: str = "TMM",
    xlabel: str = WL_UM_XLABEL,
) -> Path:
    ours_arr = np.asarray(ours, dtype=float)
    base_arr = np.asarray(baseline, dtype=float)
    if log_domain:
        ours_arr = log10_series(ours_arr)
        base_arr = log10_series(base_arr)
    out_path = out_dir / filename
    plot_1d_compare(
        x,
        ours_arr,
        base_arr,
        ylabel,
        title,
        baseline_name=baseline_name,
        xlabel=xlabel,
        tmm_label=tmm_label,
        out_path=out_path,
    )
    print(f"[plot] wrote {out_path.resolve()}", flush=True)
    return out_path


def secondary_to_primary_peak_ratio(wl: np.ndarray, spectrum: np.ndarray) -> float:
    spec = np.asarray(spectrum, dtype=float)
    if spec.size < 3 or float(np.max(spec)) <= 0.0:
        return float("nan")
    peaks: list[tuple[float, int]] = []
    for i in range(1, spec.size - 1):
        if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > 0.05 * float(np.max(spec)):
            peaks.append((float(spec[i]), i))
    if len(peaks) < 2:
        return 0.0
    peaks.sort(key=lambda item: item[0], reverse=True)
    primary, secondary = peaks[0][0], peaks[1][0]
    return secondary / primary if primary > 0.0 else float("nan")


def format_stats(stats: dict[str, float] | None) -> str:
    if not stats:
        return ""
    parts = []
    for key in ("rmse", "mae", "max_abs", "corr", "dd_corr", "peak_delta"):
        if key in stats and np.isfinite(stats[key]):
            parts.append(f"{key}={stats[key]:.4g}")
    return ", ".join(parts)


def print_case_result(result: CompareCaseResult) -> None:
    if result.skipped:
        tag = "SKIP"
    elif result.passed:
        tag = "PASS"
    else:
        tag = "FAIL"
    line = f"[{tag}] {result.name}"
    if result.message:
        line += f": {result.message}"
    stats_msg = format_stats(result.stats)
    if stats_msg:
        line += f" ({stats_msg})"
    print(line)


def plot_series(series: PlotSeries, out_dir: Path, *, plot_1d_compare) -> None:
    ours = np.asarray(series.ours, dtype=float)
    baseline = np.asarray(series.baseline, dtype=float)
    if series.log_domain:
        ours = log10_series(ours)
        baseline = log10_series(baseline)
    out_path = out_dir / series.filename if series.filename else None
    plot_1d_compare(
        series.x,
        ours,
        baseline,
        series.ylabel,
        series.title,
        baseline_name="Oghma baseline",
        xlabel=series.xlabel,
        tmm_label="ours",
        out_path=out_path,
    )


def emit_plots(result: CompareCaseResult, out_dir: Path, *, plot_1d_compare) -> None:
    if not result.plots:
        return
    for series in result.plots:
        plot_series(series, out_dir, plot_1d_compare=plot_1d_compare)


def reciprocity_visualize(
    title: str,
    baseline_key: str,
    baseline: np.ndarray,
    others: dict[str, np.ndarray],
    *,
    ignore_dc: bool = False,
) -> None:
    """Plot via compare_data; output controlled by SAVE_TO_FILE env."""
    from compare_data import compare_data, py_pipe

    keys = [baseline_key, *others.keys()]
    pipe_dict: dict[str, np.ndarray] = {baseline_key: np.asarray(baseline, dtype=float)}
    for key, data in others.items():
        pipe_dict[key] = np.asarray(data, dtype=float)
    args = Namespace(
        path=[],
        key=keys,
        amp=False,
        phase=False,
        ignore_dc=ignore_dc,
        visualize=True,
        check=[],
    )
    py_pipe.log(title)
    compare_data(args, pipe_dict=pipe_dict)


def rt_gate_failures(
    stats_r: dict,
    stats_t: dict,
    *,
    label: str = "",
    r_rmse: float = RT_R_RMSE,
    r_max: float = RT_R_MAX,
    r_corr: float = RT_R_CORR,
    t_rmse: float = RT_T_RMSE,
    t_max: float = RT_T_MAX,
) -> list[str]:
    failures: list[str] = []
    if stats_r["rmse"] >= r_rmse:
        failures.append(f"R RMSE={stats_r['rmse']:.4g} >= {r_rmse}")
    if stats_r["max_abs"] >= r_max:
        failures.append(f"R max|err|={stats_r['max_abs']:.4g} >= {r_max}")
    if stats_r["corr"] <= r_corr:
        failures.append(f"R corr={stats_r['corr']:.4f} <= {r_corr}")
    if stats_t["rmse"] >= t_rmse:
        failures.append(f"T RMSE={stats_t['rmse']:.4g} >= {t_rmse}")
    if stats_t["max_abs"] >= t_max:
        failures.append(f"T max|err|={stats_t['max_abs']:.4g} >= {t_max}")
    if label and failures:
        return [f"{label}: {msg}" for msg in failures]
    return failures


def rt_gate_passed(
    stats_r: dict,
    stats_t: dict,
    *,
    r_rmse: float = RT_R_RMSE,
    r_max: float = RT_R_MAX,
    r_corr: float = RT_R_CORR,
    t_rmse: float = RT_T_RMSE,
    t_max: float = RT_T_MAX,
) -> bool:
    return not rt_gate_failures(
        stats_r,
        stats_t,
        r_rmse=r_rmse,
        r_max=r_max,
        r_corr=r_corr,
        t_rmse=t_rmse,
        t_max=t_max,
    )


def assert_rt_gate(
    stats_r: dict,
    stats_t: dict,
    *,
    label: str,
    r_rmse: float = RT_R_RMSE,
    r_max: float = RT_R_MAX,
    r_corr: float = RT_R_CORR,
    t_rmse: float = RT_T_RMSE,
    t_max: float = RT_T_MAX,
) -> None:
    failures = rt_gate_failures(
        stats_r,
        stats_t,
        label=label,
        r_rmse=r_rmse,
        r_max=r_max,
        r_corr=r_corr,
        t_rmse=t_rmse,
        t_max=t_max,
    )
    if failures:
        pytest.fail("; ".join(failures))


def parse_oled_project_args(
    *,
    description: str,
    default_out_dir: Path,
    default_project: Path,
) -> Namespace:
    """Shared argparse for OLED alignment CLI scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=default_project,
        help="Path to Oghma 01_hello_oled project directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory for alignment plots and summaries.",
    )
    args = parser.parse_args()
    if not args.project_dir.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {args.project_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args
