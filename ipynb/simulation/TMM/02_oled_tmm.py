#!/usr/bin/env python3
"""
OLED passive/emission-stack TMM vs Oghma R/T baseline.

This script aligns:
- Passive plane-wave TMM R(λ)/T(λ) (ITO->Al illumination) vs Oghma `reflect.csv`/`transmit.csv`
- Emission-stack R(λ)/T(λ) computed via the emission-TMM API (but passive-consistent) vs the same baseline
- Cross-check: passive-path R/T ≈ emission-stack R/T
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oghma_runtime import oghma_project_dir, prepare_simulation, simulation_repo_root

_RUNTIME = prepare_simulation()
_REPO = simulation_repo_root()
import simulation  # noqa: E402
from oghma_core import compare_metrics, load_oghma_optical_reference, load_oghma_project, resolve_oghma_materials_root  # noqa: E402
from oghma_oled_utils import (
    compute_oled_emission_stack_rt_ito_al,
    compute_oled_passive_rt_ito_al,
    OLED_RT_GATES,
)
from oghma_pytest_helpers import assert_rt_gate, rt_gate_passed  # noqa: E402

DEFAULT_OLED_PROJECT = oghma_project_dir("oled", "01_hello_oled", repo=_REPO)


def _report_metrics(
    rows: list[dict],
    name: str,
    tmm: np.ndarray,
    baseline: np.ndarray,
    *,
    x: np.ndarray | None = None,
    rmse_thr: float,
    max_thr: float,
    min_corr: float | None = None,
) -> dict:
    stats = compare_metrics(tmm, baseline, x=x, peak_axis=0 if x is not None and np.ndim(tmm) == 1 else None)
    ok = bool(stats["rmse"] < rmse_thr and stats["max_abs"] < max_thr)
    if min_corr is not None and np.isfinite(stats.get("corr", np.nan)):
        ok = ok and bool(stats["corr"] >= min_corr)
    status = "PASS" if ok else "FAIL"
    row = {"case": name, **stats, "status": status}
    rows.append(row)
    print(
        f"[{status}] {name}: RMSE={stats['rmse']:.4g}, max|err|={stats['max_abs']:.4g}, "
        f"corr={stats.get('corr', float('nan')):.4f}"
    )
    return stats


def _plot_1d_compare(
    x: np.ndarray,
    y_main: np.ndarray,
    y_base: np.ndarray,
    *,
    title: str,
    ylabel: str,
    base_label: str,
    out_path: Path,
    main_label: str = "TMM",
    ref_label: str | None = None,
) -> None:
    x = np.asarray(x, dtype=float)
    y_main = np.asarray(y_main, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    err = y_main - y_base
    ref_legend = ref_label if ref_label is not None else f"{base_label} (baseline)"

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, y_main, label=main_label, lw=2)
    axes[0].plot(x, y_base, "--", label=ref_legend, lw=1.5)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, err, color="C3", lw=1.5)
    axes[1].axhline(0.0, color="k", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("λ (μm)")
    axes[1].set_ylabel("Δ")
    axes[1].grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _band_max_abs(x_um: np.ndarray, err: np.ndarray, lo_um: float, hi_um: float) -> float:
    x_um = np.asarray(x_um, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = (x_um >= float(lo_um)) & (x_um <= float(hi_um))
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(err[mask])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=DEFAULT_OLED_PROJECT)
    p.add_argument("--out-dir", type=Path, default=Path("./output/02_oled_tmm"))
    p.add_argument("--incident-angle-deg", type=float, default=0.0)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    project = load_oghma_project(args.project)
    ref = load_oghma_optical_reference(project)
    wl_um = np.asarray(ref["reflect_wl_um"], dtype=float)
    r_base = np.asarray(ref["reflect"], dtype=float)
    t_base = np.asarray(ref["transmit"], dtype=float)
    theta_rad = float(args.incident_angle_deg) * np.pi / 180.0

    print(f"RUNTIME={_RUNTIME}")
    print(f"TMM_DIR={Path(__file__).resolve().parent}")
    print(f"oghma_database/materials: {resolve_oghma_materials_root()}")
    print(f"project: {args.project}")
    print(f"simmode: {project.simmode}, thickness: {project.total_thickness_um:.3f} μm")
    print(f"λ: {wl_um[0]:.3f}–{wl_um[-1]:.3f} μm ({len(wl_um)} pts), θ={args.incident_angle_deg:g}°")

    alignment: list[dict] = []

    r_pass, t_pass = compute_oled_passive_rt_ito_al(
        project,
        wl_um,
        simulation,
        incident_angle_rad=theta_rad,
    )
    r_emis, t_emis = compute_oled_emission_stack_rt_ito_al(
        project,
        wl_um,
        simulation,
        incident_angle_rad=theta_rad,
    )

    stats_r_pass = _report_metrics(
        alignment,
        "R(λ) passive vs baseline",
        r_pass,
        r_base,
        x=wl_um,
        rmse_thr=OLED_RT_GATES["r_rmse"],
        max_thr=OLED_RT_GATES["r_max"],
        min_corr=OLED_RT_GATES["r_corr"],
    )
    stats_t_pass = _report_metrics(
        alignment,
        "T(λ) passive vs baseline",
        t_pass,
        t_base,
        x=wl_um,
        rmse_thr=OLED_RT_GATES["t_rmse"],
        max_thr=OLED_RT_GATES["t_max"],
        min_corr=None,
    )
    stats_r_emis = _report_metrics(
        alignment,
        "R(λ) emission-stack vs baseline",
        r_emis,
        r_base,
        x=wl_um,
        rmse_thr=OLED_RT_GATES["r_rmse"],
        max_thr=OLED_RT_GATES["r_max"],
        min_corr=OLED_RT_GATES["r_corr"],
    )
    stats_t_emis = _report_metrics(
        alignment,
        "T(λ) emission-stack vs baseline",
        t_emis,
        t_base,
        x=wl_um,
        rmse_thr=OLED_RT_GATES["t_rmse"],
        max_thr=OLED_RT_GATES["t_max"],
        min_corr=None,
    )

    # --- Cross-path consistency (passive ≈ emission-stack)
    cross_err_r = float(np.max(np.abs(np.asarray(r_pass) - np.asarray(r_emis))))
    cross_err_t = float(np.max(np.abs(np.asarray(t_pass) - np.asarray(t_emis))))
    alignment.append(
        {
            "case": "cross passive≈emission-stack (max|Δ|)",
            "rmse": float("nan"),
            "max_abs": max(cross_err_r, cross_err_t),
            "corr": float("nan"),
            "status": "INFO",
        }
    )

    # --- Plots
    _plot_1d_compare(
        wl_um,
        r_pass,
        r_base,
        title="Reflection spectrum (passive TMM vs Oghma)",
        ylabel="R",
        base_label="R",
        out_path=out_dir / "rt_reflection.png",
    )
    _plot_1d_compare(
        wl_um,
        t_pass,
        t_base,
        title="Transmission spectrum (passive TMM vs Oghma)",
        ylabel="T",
        base_label="T",
        out_path=out_dir / "rt_transmission.png",
    )
    _plot_1d_compare(
        wl_um,
        r_emis,
        r_pass,
        title="Reflection spectrum (emission TMM vs passive TMM)",
        ylabel="R",
        base_label="R",
        main_label="emission TMM",
        ref_label="passive TMM",
        out_path=out_dir / "rt_passive_emission_reflection.png",
    )
    _plot_1d_compare(
        wl_um,
        t_emis,
        t_pass,
        title="Transmission spectrum (emission TMM vs passive TMM)",
        ylabel="T",
        base_label="T",
        main_label="emission TMM",
        ref_label="passive TMM",
        out_path=out_dir / "rt_passive_emission_transmission.png",
    )

    r_err = np.asarray(r_pass, dtype=float) - np.asarray(r_base, dtype=float)
    r_300_350 = _band_max_abs(wl_um, r_err, 0.30, 0.35)
    print(f"Band max|ΔR| @300–350nm: {r_300_350:.4g}")

    # --- Strict gates (for CI/automation; match 02_oled_tmm.ipynb RT_GATE_PASSED)
    failures: list[str] = []
    if not rt_gate_passed(stats_r_pass, stats_t_pass, **OLED_RT_GATES):
        failures.append("passive RT gate failed")
    if not rt_gate_passed(stats_r_emis, stats_t_emis, **OLED_RT_GATES):
        failures.append("emission-stack RT gate failed")
    if not (r_300_350 < OLED_RT_GATES["r_max"]):
        failures.append("300-350nm R max|Δ| gate failed")

    # --- Report
    df = pd.DataFrame(alignment)
    report_path = out_dir / "alignment_report.csv"
    df.to_csv(report_path, index=False)
    print(df.to_string(index=False))
    print(f"Wrote plots and {report_path} to {out_dir.resolve()}")
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

