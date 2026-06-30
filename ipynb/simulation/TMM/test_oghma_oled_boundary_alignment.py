#!/usr/bin/env python3
"""OLED stack boundary R/T alignment: ito_al vs air bookends vs Oghma baseline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oghma_runtime import (
    bootstrap_tmm_session,
    default_oled_hello_project,
    tmm_output_dir,
)

_, _RUNTIME = bootstrap_tmm_session(import_tmm=False)
import simulation  # noqa: E402
from oghma_core import compare_metrics, load_oghma_optical_reference, load_oghma_project  # noqa: E402
from oghma_oled_utils import (  # noqa: E402
    OLED_RT_GATES,
    build_oghma_oled_emission_stack_ito_al_air_boundary,
    build_oghma_oled_passive_stack_ito_al,
    compute_oled_emission_stack_rt,
    compute_oled_stack_rt_power_unpolarized,
)
from oghma_fdtd_alignment import plot_1d_compare  # noqa: E402
from oghma_pytest_helpers import (  # noqa: E402
    CROSS_PATH_RTOL,
    assert_rt_gate,
    log_before_visualize,
    reciprocity_visualize,
)

DEFAULT_OUT_DIR = tmm_output_dir("test_oghma_oled_boundary_alignment")

ITO_AL_LABEL = "ITO/Al bookends"
AIR_LABEL = "air bookends (n=1)"


@dataclass(frozen=True)
class BoundaryRTSpectra:
    r_ito_al_passive: np.ndarray
    t_ito_al_passive: np.ndarray
    r_ito_al_emission: np.ndarray
    t_ito_al_emission: np.ndarray
    r_air_passive: np.ndarray
    t_air_passive: np.ndarray
    r_air_emission: np.ndarray
    t_air_emission: np.ndarray


@dataclass(frozen=True)
class BoundaryRTStats:
    ito_al_passive_r: dict[str, float]
    ito_al_passive_t: dict[str, float]
    ito_al_emission_r: dict[str, float]
    ito_al_emission_t: dict[str, float]
    air_passive_r: dict[str, float]
    air_passive_t: dict[str, float]
    air_emission_r: dict[str, float]
    air_emission_t: dict[str, float]



def _compute_all_boundary_rt(
    project,
    wl_um: np.ndarray,
    sim,
) -> BoundaryRTSpectra:
    r_ito_al_passive, t_ito_al_passive = compute_oled_stack_rt_power_unpolarized(
        project,
        wl_um,
        sim,
        stack_builder=build_oghma_oled_passive_stack_ito_al,
    )
    r_ito_al_emission, t_ito_al_emission = compute_oled_emission_stack_rt(
        project,
        wl_um,
        sim,
        stack_builder=build_oghma_oled_passive_stack_ito_al,
    )
    r_air_passive, t_air_passive = compute_oled_stack_rt_power_unpolarized(
        project,
        wl_um,
        sim,
        stack_builder=build_oghma_oled_emission_stack_ito_al_air_boundary,
    )
    r_air_emission, t_air_emission = compute_oled_emission_stack_rt(
        project,
        wl_um,
        sim,
        stack_builder=build_oghma_oled_emission_stack_ito_al_air_boundary,
    )
    return BoundaryRTSpectra(
        r_ito_al_passive=r_ito_al_passive,
        t_ito_al_passive=t_ito_al_passive,
        r_ito_al_emission=r_ito_al_emission,
        t_ito_al_emission=t_ito_al_emission,
        r_air_passive=r_air_passive,
        t_air_passive=t_air_passive,
        r_air_emission=r_air_emission,
        t_air_emission=t_air_emission,
    )


def _compute_all_boundary_stats(
    spectra: BoundaryRTSpectra,
    r_base: np.ndarray,
    t_base: np.ndarray,
    wl_um: np.ndarray,
) -> BoundaryRTStats:
    return BoundaryRTStats(
        ito_al_passive_r=compare_metrics(
            spectra.r_ito_al_passive, r_base, x=wl_um, peak_axis=0
        ),
        ito_al_passive_t=compare_metrics(
            spectra.t_ito_al_passive, t_base, x=wl_um, peak_axis=0
        ),
        ito_al_emission_r=compare_metrics(
            spectra.r_ito_al_emission, r_base, x=wl_um, peak_axis=0
        ),
        ito_al_emission_t=compare_metrics(
            spectra.t_ito_al_emission, t_base, x=wl_um, peak_axis=0
        ),
        air_passive_r=compare_metrics(
            spectra.r_air_passive, r_base, x=wl_um, peak_axis=0
        ),
        air_passive_t=compare_metrics(
            spectra.t_air_passive, t_base, x=wl_um, peak_axis=0
        ),
        air_emission_r=compare_metrics(
            spectra.r_air_emission, r_base, x=wl_um, peak_axis=0
        ),
        air_emission_t=compare_metrics(
            spectra.t_air_emission, t_base, x=wl_um, peak_axis=0
        ),
    )


def _format_stats_table(stats: BoundaryRTStats) -> str:
    rows = (
        ("ito_al", "passive", "R", stats.ito_al_passive_r),
        ("ito_al", "passive", "T", stats.ito_al_passive_t),
        ("ito_al", "emission", "R", stats.ito_al_emission_r),
        ("ito_al", "emission", "T", stats.ito_al_emission_t),
        ("air", "passive", "R", stats.air_passive_r),
        ("air", "passive", "T", stats.air_passive_t),
        ("air", "emission", "R", stats.air_emission_r),
        ("air", "emission", "T", stats.air_emission_t),
    )
    lines = [
        f"{'boundary':<10} {'solver':<10} {'metric':<6} {'rmse':>10} {'max_abs':>10} {'corr':>8}",
        "-" * 58,
    ]
    for boundary, solver, metric, s in rows:
        lines.append(
            f"{boundary:<10} {solver:<10} {metric:<6} "
            f"{s['rmse']:10.6g} {s['max_abs']:10.6g} {s['corr']:8.4f}"
        )
    return "\n".join(lines)


def _plot_rt_vs_baseline(
    wl_um: np.ndarray,
    ours: np.ndarray,
    baseline: np.ndarray,
    *,
    metric: str,
    title: str,
    filename: str,
    out_dir: Path,
    tmm_label: str,
    baseline_name: str = "Oghma baseline",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plot_1d_compare(
        np.asarray(wl_um, dtype=float),
        np.asarray(ours, dtype=float),
        np.asarray(baseline, dtype=float),
        metric,
        title,
        baseline_name=baseline_name,
        tmm_label=tmm_label,
        out_path=out_path,
    )
    print(f"[plot] wrote {out_path.resolve()}", flush=True)
    return out_path


def _write_all_case_plots(
    spectra: BoundaryRTSpectra,
    stats: BoundaryRTStats,
    wl_um: np.ndarray,
    r_base: np.ndarray,
    t_base: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)

    _plot_rt_vs_baseline(
        wl_um,
        spectra.r_ito_al_passive,
        r_base,
        metric="R",
        title=(
            f"R(λ) — {ITO_AL_LABEL}, passive TMM vs Oghma baseline "
            f"(rmse={stats.ito_al_passive_r['rmse']:.4g}, "
            f"max|err|={stats.ito_al_passive_r['max_abs']:.4g}, "
            f"corr={stats.ito_al_passive_r['corr']:.4f})"
        ),
        filename="ito_al_passive_R.png",
        out_dir=out_dir,
        tmm_label="ito_al passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.t_ito_al_passive,
        t_base,
        metric="T",
        title=(
            f"T(λ) — {ITO_AL_LABEL}, passive TMM vs Oghma baseline "
            f"(rmse={stats.ito_al_passive_t['rmse']:.4g}, "
            f"max|err|={stats.ito_al_passive_t['max_abs']:.4g}, "
            f"corr={stats.ito_al_passive_t['corr']:.4f})"
        ),
        filename="ito_al_passive_T.png",
        out_dir=out_dir,
        tmm_label="ito_al passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.r_ito_al_emission,
        r_base,
        metric="R",
        title=(
            f"R(λ) — {ITO_AL_LABEL}, emission TMM vs Oghma baseline "
            f"(rmse={stats.ito_al_emission_r['rmse']:.4g}, "
            f"max|err|={stats.ito_al_emission_r['max_abs']:.4g}, "
            f"corr={stats.ito_al_emission_r['corr']:.4f})"
        ),
        filename="ito_al_emission_R.png",
        out_dir=out_dir,
        tmm_label="ito_al emission TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.t_ito_al_emission,
        t_base,
        metric="T",
        title=(
            f"T(λ) — {ITO_AL_LABEL}, emission TMM vs Oghma baseline "
            f"(rmse={stats.ito_al_emission_t['rmse']:.4g}, "
            f"max|err|={stats.ito_al_emission_t['max_abs']:.4g}, "
            f"corr={stats.ito_al_emission_t['corr']:.4f})"
        ),
        filename="ito_al_emission_T.png",
        out_dir=out_dir,
        tmm_label="ito_al emission TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.r_air_passive,
        r_base,
        metric="R",
        title=(
            f"R(λ) — {AIR_LABEL}, passive TMM vs Oghma baseline "
            f"(rmse={stats.air_passive_r['rmse']:.4g}, "
            f"max|err|={stats.air_passive_r['max_abs']:.4g}, "
            f"corr={stats.air_passive_r['corr']:.4f})"
        ),
        filename="air_passive_R.png",
        out_dir=out_dir,
        tmm_label="air passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.t_air_passive,
        t_base,
        metric="T",
        title=(
            f"T(λ) — {AIR_LABEL}, passive TMM vs Oghma baseline "
            f"(rmse={stats.air_passive_t['rmse']:.4g}, "
            f"max|err|={stats.air_passive_t['max_abs']:.4g}, "
            f"corr={stats.air_passive_t['corr']:.4f})"
        ),
        filename="air_passive_T.png",
        out_dir=out_dir,
        tmm_label="air passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.r_air_emission,
        r_base,
        metric="R",
        title=(
            f"R(λ) — {AIR_LABEL}, emission TMM vs Oghma baseline "
            f"(rmse={stats.air_emission_r['rmse']:.4g}, "
            f"max|err|={stats.air_emission_r['max_abs']:.4g}, "
            f"corr={stats.air_emission_r['corr']:.4f})"
        ),
        filename="air_emission_R.png",
        out_dir=out_dir,
        tmm_label="air emission TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.t_air_emission,
        t_base,
        metric="T",
        title=(
            f"T(λ) — {AIR_LABEL}, emission TMM vs Oghma baseline "
            f"(rmse={stats.air_emission_t['rmse']:.4g}, "
            f"max|err|={stats.air_emission_t['max_abs']:.4g}, "
            f"corr={stats.air_emission_t['corr']:.4f})"
        ),
        filename="air_emission_T.png",
        out_dir=out_dir,
        tmm_label="air emission TMM",
    )


def _write_cross_path_plots(
    spectra: BoundaryRTSpectra,
    wl_um: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)

    _plot_rt_vs_baseline(
        wl_um,
        spectra.r_ito_al_emission,
        spectra.r_ito_al_passive,
        metric="R",
        title=f"R(λ) — {ITO_AL_LABEL}: passive TMM vs emission TMM",
        filename="ito_al_R_passive_vs_emission.png",
        out_dir=out_dir,
        tmm_label="ito_al emission TMM",
        baseline_name="ito_al passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.t_ito_al_emission,
        spectra.t_ito_al_passive,
        metric="T",
        title=f"T(λ) — {ITO_AL_LABEL}: passive TMM vs emission TMM",
        filename="ito_al_T_passive_vs_emission.png",
        out_dir=out_dir,
        tmm_label="ito_al emission TMM",
        baseline_name="ito_al passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.r_air_emission,
        spectra.r_air_passive,
        metric="R",
        title=f"R(λ) — {AIR_LABEL}: passive TMM vs emission TMM",
        filename="air_R_passive_vs_emission.png",
        out_dir=out_dir,
        tmm_label="air emission TMM",
        baseline_name="air passive TMM",
    )
    _plot_rt_vs_baseline(
        wl_um,
        spectra.t_air_emission,
        spectra.t_air_passive,
        metric="T",
        title=f"T(λ) — {AIR_LABEL}: passive TMM vs emission TMM",
        filename="air_T_passive_vs_emission.png",
        out_dir=out_dir,
        tmm_label="air emission TMM",
        baseline_name="air passive TMM",
    )


def _write_summary_txt(stats: BoundaryRTStats, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "boundary_rmse_summary.txt"
    txt_path.write_text(_format_stats_table(stats) + "\n", encoding="utf-8")
    print(f"[summary] wrote {txt_path.resolve()}", flush=True)
    return txt_path


def _visualize_boundary_rt(
    spectra: BoundaryRTSpectra,
    stats: BoundaryRTStats,
    r_base: np.ndarray,
    t_base: np.ndarray,
    wl_um: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    _write_all_case_plots(spectra, stats, wl_um, r_base, t_base, out_dir)
    _write_cross_path_plots(spectra, wl_um, out_dir)

    wl_arr = np.asarray(wl_um, dtype=float)
    log_before_visualize(
        "R(λ) — ito_al / air × passive / emission vs Oghma reflect.csv",
        n_wl=len(wl_arr),
        wl_range=f"{wl_arr[0]:.3f}–{wl_arr[-1]:.3f} μm",
    )
    reciprocity_visualize(
        "oled_boundary_R_all_paths",
        "r_oghma_baseline",
        r_base,
        {
            "r_ito_al_passive": spectra.r_ito_al_passive,
            "r_ito_al_emission": spectra.r_ito_al_emission,
            "r_air_passive": spectra.r_air_passive,
            "r_air_emission": spectra.r_air_emission,
        },
    )
    log_before_visualize(
        "T(λ) — ito_al / air × passive / emission vs Oghma transmit.csv",
        n_wl=len(wl_arr),
        wl_range=f"{wl_arr[0]:.3f}–{wl_arr[-1]:.3f} μm",
    )
    reciprocity_visualize(
        "oled_boundary_T_all_paths",
        "t_oghma_baseline",
        t_base,
        {
            "t_ito_al_passive": spectra.t_ito_al_passive,
            "t_ito_al_emission": spectra.t_ito_al_emission,
            "t_air_passive": spectra.t_air_passive,
            "t_air_emission": spectra.t_air_emission,
        },
    )


def _load_project_context(project_dir: Path):
    project = load_oghma_project(project_dir)
    ref = load_oghma_optical_reference(project)
    wl_um = np.asarray(ref["reflect_wl_um"], dtype=float)
    r_base = np.asarray(ref["reflect"], dtype=float)
    t_base = np.asarray(ref["transmit"], dtype=float)
    return project, wl_um, r_base, t_base


def _assert_finite_stats(s: dict[str, float]) -> None:
    assert np.isfinite(s["rmse"])
    assert np.isfinite(s["max_abs"])
    assert np.isfinite(s["corr"])


def run_oled_boundary_rt_matrix_vs_baseline(
    sim,
    project_dir: Path,
    *,
    out_dir: Path | None = None,
):
    """Compute 2×2×2 R/T spectra and compare each to Oghma reflect/transmit baseline."""
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    project, wl_um, r_base, t_base = _load_project_context(project_dir)
    spectra = _compute_all_boundary_rt(project, wl_um, sim)
    stats = _compute_all_boundary_stats(spectra, r_base, t_base, wl_um)
    _write_all_case_plots(spectra, stats, wl_um, r_base, t_base, out_dir)

    _assert_finite_stats(stats.ito_al_passive_r)
    _assert_finite_stats(stats.ito_al_passive_t)
    _assert_finite_stats(stats.ito_al_emission_r)
    _assert_finite_stats(stats.ito_al_emission_t)
    _assert_finite_stats(stats.air_passive_r)
    _assert_finite_stats(stats.air_passive_t)
    _assert_finite_stats(stats.air_emission_r)
    _assert_finite_stats(stats.air_emission_t)


def run_oled_ito_al_rt_gates(
    sim,
    project_dir: Path,
    *,
    out_dir: Path | None = None,
):
    """ito_al passive/emission R/T must pass Oghma gates; paths agree within CROSS_PATH_RTOL."""
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    project, wl_um, r_base, t_base = _load_project_context(project_dir)
    spectra = _compute_all_boundary_rt(project, wl_um, sim)
    _write_cross_path_plots(spectra, wl_um, out_dir)

    np.testing.assert_allclose(
        spectra.r_ito_al_passive,
        spectra.r_ito_al_emission,
        rtol=0.0,
        atol=CROSS_PATH_RTOL,
    )
    np.testing.assert_allclose(
        spectra.t_ito_al_passive,
        spectra.t_ito_al_emission,
        rtol=0.0,
        atol=CROSS_PATH_RTOL,
    )

    assert_rt_gate(
        compare_metrics(spectra.r_ito_al_passive, r_base, x=wl_um, peak_axis=0),
        compare_metrics(spectra.t_ito_al_passive, t_base, x=wl_um, peak_axis=0),
        label="ito_al passive TMM",
        **OLED_RT_GATES,
    )
    assert_rt_gate(
        compare_metrics(spectra.r_ito_al_emission, r_base, x=wl_um, peak_axis=0),
        compare_metrics(spectra.t_ito_al_emission, t_base, x=wl_um, peak_axis=0),
        label="ito_al emission TMM",
        **OLED_RT_GATES,
    )


def run_oled_boundary_closer_to_baseline(
    sim,
    project_dir: Path,
    *,
    out_dir: Path | None = None,
):
    """Report which bookend boundary is closer to Oghma R/T baseline (diagnostic)."""
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR
    project, wl_um, r_base, t_base = _load_project_context(project_dir)
    spectra = _compute_all_boundary_rt(project, wl_um, sim)
    stats = _compute_all_boundary_stats(spectra, r_base, t_base, wl_um)

    table = _format_stats_table(stats)
    print("\n[boundary RT vs Oghma baseline]\n" + table + "\n", flush=True)

    _write_summary_txt(stats, Path(out_dir))
    _visualize_boundary_rt(spectra, stats, r_base, t_base, wl_um, Path(out_dir))

    print(
        "[compare] R passive: "
        f"ito_al rmse={stats.ito_al_passive_r['rmse']:.6g} "
        f"max_abs={stats.ito_al_passive_r['max_abs']:.6g} | "
        f"air rmse={stats.air_passive_r['rmse']:.6g} "
        f"max_abs={stats.air_passive_r['max_abs']:.6g} | "
        f"closer={'ito_al' if stats.ito_al_passive_r['rmse'] <= stats.air_passive_r['rmse'] else 'air'}",
        flush=True,
    )
    print(
        "[compare] R emission: "
        f"ito_al rmse={stats.ito_al_emission_r['rmse']:.6g} "
        f"max_abs={stats.ito_al_emission_r['max_abs']:.6g} | "
        f"air rmse={stats.air_emission_r['rmse']:.6g} "
        f"max_abs={stats.air_emission_r['max_abs']:.6g} | "
        f"closer={'ito_al' if stats.ito_al_emission_r['rmse'] <= stats.air_emission_r['rmse'] else 'air'}",
        flush=True,
    )
    print(
        "[compare] T passive: "
        f"ito_al rmse={stats.ito_al_passive_t['rmse']:.6g} "
        f"max_abs={stats.ito_al_passive_t['max_abs']:.6g} | "
        f"air rmse={stats.air_passive_t['rmse']:.6g} "
        f"max_abs={stats.air_passive_t['max_abs']:.6g} | "
        f"closer={'ito_al' if stats.ito_al_passive_t['rmse'] <= stats.air_passive_t['rmse'] else 'air'}",
        flush=True,
    )
    print(
        "[compare] T emission: "
        f"ito_al rmse={stats.ito_al_emission_t['rmse']:.6g} "
        f"max_abs={stats.ito_al_emission_t['max_abs']:.6g} | "
        f"air rmse={stats.air_emission_t['rmse']:.6g} "
        f"max_abs={stats.air_emission_t['max_abs']:.6g} | "
        f"closer={'ito_al' if stats.ito_al_emission_t['rmse'] <= stats.air_emission_t['rmse'] else 'air'}",
        flush=True,
    )

    np.testing.assert_allclose(
        spectra.r_air_passive,
        spectra.r_air_emission,
        rtol=0.0,
        atol=CROSS_PATH_RTOL,
    )
    np.testing.assert_allclose(
        spectra.t_air_passive,
        spectra.t_air_emission,
        rtol=0.0,
        atol=CROSS_PATH_RTOL,
    )

    if stats.air_passive_r["rmse"] < stats.ito_al_passive_r["rmse"]:
        print(
            "[note] air bookends closer than ito_al for R passive "
            f"(air rmse={stats.air_passive_r['rmse']:.6g} "
            f"< ito_al rmse={stats.ito_al_passive_r['rmse']:.6g})",
            flush=True,
        )
    if stats.air_emission_r["rmse"] < stats.ito_al_emission_r["rmse"]:
        print(
            "[note] air bookends closer than ito_al for R emission "
            f"(air rmse={stats.air_emission_r['rmse']:.6g} "
            f"< ito_al rmse={stats.ito_al_emission_r['rmse']:.6g})",
            flush=True,
        )
    if stats.air_passive_t["rmse"] < stats.ito_al_passive_t["rmse"]:
        print(
            "[note] air bookends closer than ito_al for T passive "
            f"(air rmse={stats.air_passive_t['rmse']:.6g} "
            f"< ito_al rmse={stats.ito_al_passive_t['rmse']:.6g})",
            flush=True,
        )
    if stats.air_emission_t["rmse"] < stats.ito_al_emission_t["rmse"]:
        print(
            "[note] air bookends closer than ito_al for T emission "
            f"(air rmse={stats.air_emission_t['rmse']:.6g} "
            f"< ito_al rmse={stats.ito_al_emission_t['rmse']:.6g})",
            flush=True,
        )


def main() -> int:
    from oghma_pytest_helpers import parse_oled_project_args

    args = parse_oled_project_args(
        description="Run OLED boundary R/T alignment checks (ito_al vs air bookends).",
        default_out_dir=DEFAULT_OUT_DIR,
        default_project=default_oled_hello_project(),
    )
    project_dir = args.project_dir

    print(f"RUNTIME={_RUNTIME}")
    print(f"Running 3 checks for project: {project_dir}")

    print("[RUN ] oled_boundary_rt_matrix_vs_baseline")
    run_oled_boundary_rt_matrix_vs_baseline(simulation, project_dir, out_dir=args.out_dir)
    print("[ OK ] oled_boundary_rt_matrix_vs_baseline")

    print("[RUN ] oled_ito_al_rt_gates")
    run_oled_ito_al_rt_gates(simulation, project_dir, out_dir=args.out_dir)
    print("[ OK ] oled_ito_al_rt_gates")

    print("[RUN ] oled_boundary_closer_to_baseline")
    run_oled_boundary_closer_to_baseline(simulation, project_dir, out_dir=args.out_dir)
    print("[ OK ] oled_boundary_closer_to_baseline")

    print(f"Wrote plots to {args.out_dir.resolve()}")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
