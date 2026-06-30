#!/usr/bin/env python3
"""Oghma Bragg / Fabry-Pérot FDTD project alignment vs TMM helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from oghma_runtime import bootstrap_tmm_session, oghma_project_dir

_TMM = Path(__file__).resolve().parent
_, _RUNTIME = bootstrap_tmm_session(import_tmm=False)

from oghma_fdtd_alignment import plot_1d_compare  # noqa: E402
from oghma_pytest_helpers import log_before_visualize, plot_compare_1d, wl_grid_um  # noqa: E402
from oghma_bragg import (  # noqa: E402
    compute_bragg_emission_transfer,
    compute_bragg_lam_e_norm,
    compute_bragg_passive_transfer,
    compute_bragg_reflection_ahead_of_z,
    parse_bragg_grating_geometry,
)
from oghma_core import _cplx_from_py, compare_metrics, fdtd_input_combine_power, lam_e_norm_from_transfer  # noqa: E402
from oghma_fabry import (  # noqa: E402
    build_fabry_perot_fdtd_stack,
    compute_fabry_lam_e_norm,
    compute_fabry_passive_transfer,
    compute_fabry_reflection_ahead_of_z,
    parse_fabry_perot_geometry,
)
from oghma_fdtd import parse_fdtd_layout  # noqa: E402
from oghma_fdtd_source import (  # noqa: E402
    compute_fdtd_source_spectrum_i_new,
    compute_fdtd_steady_e_field_at_z,
    parse_fdtd_light_source_config,
)

BRAGG_CHECKS = (
    "bragg_lam_e_norm",
    "bragg_emission_trans_ratio_vs_passive_t",
    "bragg_emission_fdtd_trans_ratio",
)

FABRY_CHECKS = (
    "fdtd_steady_i_new_cancels_in_norm",
    "fdtd_input_combine_fabry_gin",
    "fdtd_steady_field_ratio",
)


def _bragg_sim_path() -> Path:
    return oghma_project_dir("bragg_grating", "01_hello_bragg_grating") / "sim.json"


def _fabry_sim_path() -> Path:
    return oghma_project_dir("fabry_perot", "01_hello_fabry_perot") / "sim.json"


def _load_simulation_module():
    import simulation  # noqa: WPS433

    return simulation


def run_bragg_lam_e_norm(sim, *, out_dir: Path) -> None:
    """Bragg lam_E_norm helper reuses passive T and fdtd_input_combine_power G_in."""
    sim_path = _bragg_sim_path()
    geom = parse_bragg_grating_geometry(sim_path)
    layout = parse_fdtd_layout(sim_path)
    wl_um, wl_nm = wl_grid_um(430.0, 720.0, 40)
    t_passive = compute_bragg_passive_transfer(
        geom, wl_um, sim
    )
    r_coeff = compute_bragg_reflection_ahead_of_z(
        layout.z_detector_in_um,
        geom,
        layout,
        wl_um,
        sim,
    )
    expected = lam_e_norm_from_transfer(t_passive, r_coeff)
    ratio = compute_bragg_lam_e_norm(
        geom, layout, wl_um, sim
    )
    max_abs = float(np.max(np.abs(ratio - expected)))
    log_before_visualize(
        "Bragg lam_E_norm — expected T/G_in vs computed helper",
        max_abs=f"{max_abs:.6g}",
        wl_range=f"{wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm",
    )
    plot_compare_1d(
        wl_nm,
        ratio,
        expected,
        ylabel="lam_E_norm",
        title="Bragg lam_E_norm: computed vs expected",
        filename="bragg_lam_e_norm.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        baseline_name="expected",
        tmm_label="computed",
    )
    assert np.allclose(ratio, expected, rtol=1e-6, atol=1e-12)
    assert np.allclose(ratio, t_passive / fdtd_input_combine_power(r_coeff), rtol=1e-6, atol=1e-12)


def run_bragg_emission_trans_ratio_vs_passive_t(sim, *, out_dir: Path) -> None:
    """Bragg stack: P_bot/(P_top+P_bot) spectral shape tracks passive T(λ)."""
    sim_path = _bragg_sim_path()
    geom = parse_bragg_grating_geometry(sim_path)
    layout = parse_fdtd_layout(sim_path)
    wl_um, wl_nm = wl_grid_um(430.0, 720.0, 40)

    em = compute_bragg_emission_transfer(
        geom,
        layout,
        wl_um,
        sim,
        z_um=layout.z_source_um,
        u=0.0,
        orient="hs",
    )
    t_passive = compute_bragg_passive_transfer(
        geom, wl_um, sim
    )
    t_em = em["t_trans_ratio"]
    corr = float(np.corrcoef(
        np.log10(np.maximum(t_em, 1e-30)),
        np.log10(np.maximum(t_passive, 1e-30)),
    )[0, 1])
    log_before_visualize(
        "Bragg log10(t_trans_ratio) vs log10(T_passive)",
        corr=f"{corr:.4f}",
        wl_range=f"{wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm",
    )
    plot_compare_1d(
        wl_nm,
        t_em,
        t_passive,
        ylabel="log10(value)",
        title="Bragg log10(t_trans_ratio) vs log10(T_passive)",
        filename="bragg_log_t_trans_vs_passive_T.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        log_domain=True,
        baseline_name="log10(T_passive)",
        tmm_label="log10(t_trans_ratio)",
    )
    assert corr > 0.90, f"Bragg log t_trans vs log passive T correlation {corr:.4f} < 0.90"


def run_bragg_emission_fdtd_trans_ratio(sim, *, out_dir: Path) -> None:
    """Case A: P_bot/(G_in·(P_top+P_bot)) tracks T_passive/G_in in Bragg window."""
    sim_path = _bragg_sim_path()
    geom = parse_bragg_grating_geometry(sim_path)
    layout = parse_fdtd_layout(sim_path)
    wl_um, wl_nm = wl_grid_um(430.0, 720.0, 40)
    em = compute_bragg_emission_transfer(
        geom,
        layout,
        wl_um,
        sim,
        z_um=layout.z_source_um,
        u=0.0,
        orient="hs",
    )
    t_base = compute_bragg_lam_e_norm(
        geom, layout, wl_um, sim
    )
    t_fdtd = em["t_trans_ratio_fdtd"]
    stats = compare_metrics(t_fdtd, t_base, x=wl_um)
    log_before_visualize(
        "Bragg Case A — t_trans_ratio_fdtd vs lam_e_norm",
        corr=f"{stats['corr']:.4f}",
        rmse=f"{stats['rmse']:.4g}",
        max_abs=f"{stats['max_abs']:.4g}",
    )
    plot_compare_1d(
        wl_nm,
        t_fdtd,
        t_base,
        ylabel="transfer ratio",
        title="Bragg Case A: t_trans_ratio_fdtd vs lam_e_norm",
        filename="bragg_fdtd_trans_vs_lam_e_norm.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        baseline_name="lam_e_norm",
        tmm_label="t_trans_ratio_fdtd",
    )
    assert stats["corr"] >= 0.80, f"Case A Bragg corr {stats['corr']:.4f} < 0.80"


def run_fdtd_steady_i_new_cancels_in_norm(sim, *, out_dir: Path) -> None:
    """Scaling I_new changes |E|² but not |E_out|²/|E_in|²."""
    sim_path = _fabry_sim_path()
    geom = parse_fabry_perot_geometry(sim_path)
    layout = parse_fdtd_layout(sim_path)
    src_cfg = parse_fdtd_light_source_config(sim_path)
    wl_um, wl_nm = wl_grid_um(480.0, 520.0, 12)
    i_new = compute_fdtd_source_spectrum_i_new(wl_um, src_cfg)
    scale = 4.0

    def build_layers(wl_val: float):
        return build_fabry_perot_fdtd_stack(
            geom, layout, wl_val, sim
        )

    def _ratio(i_arr: np.ndarray) -> np.ndarray:
        e_in = compute_fdtd_steady_e_field_at_z(
            wl_um,
            src_cfg,
            layout.z_detector_in_um,
            build_layers,
            sim,
            i_new=i_arr,
        )
        e_out = compute_fdtd_steady_e_field_at_z(
            wl_um,
            src_cfg,
            layout.z_detector_out_um,
            build_layers,
            sim,
            i_new=i_arr,
        )
        return np.abs(e_out) ** 2 / np.maximum(np.abs(e_in) ** 2, 1e-30)

    ratio_base = _ratio(i_new)
    ratio_scaled = _ratio(scale * i_new)
    ratio_max_diff = float(np.max(np.abs(ratio_base - ratio_scaled)))
    log_before_visualize(
        "Fabry I_new scaling — |E_out|²/|E_in|² invariance under 4× I_new",
        scale=scale,
        max_abs_ratio_diff=f"{ratio_max_diff:.6g}",
        wl_range=f"{wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm",
    )
    plot_compare_1d(
        wl_nm,
        ratio_scaled,
        ratio_base,
        ylabel="|E_out|²/|E_in|²",
        title="Fabry I_new ratio invariance (base vs 4× I_new)",
        filename="fabry_i_new_ratio_invariance.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        baseline_name="ratio (I_new)",
        tmm_label="ratio (4× I_new)",
    )
    assert np.allclose(ratio_base, ratio_scaled, rtol=1e-12, atol=1e-12)

    e_in_base = compute_fdtd_steady_e_field_at_z(
        wl_um,
        src_cfg,
        layout.z_detector_in_um,
        build_layers,
        sim,
        i_new=i_new,
    )
    e_in_scaled = compute_fdtd_steady_e_field_at_z(
        wl_um,
        src_cfg,
        layout.z_detector_in_um,
        build_layers,
        sim,
        i_new=scale * i_new,
    )
    e_in_base_pow = np.abs(e_in_base) ** 2
    e_in_scaled_pow = np.abs(e_in_scaled) ** 2
    e_in_expected = scale * e_in_base_pow
    log_before_visualize(
        "Fabry I_new scaling — |E_in|² scales linearly with I_new",
        scale=scale,
        max_abs_diff=f"{float(np.max(np.abs(e_in_scaled_pow - e_in_expected))):.6g}",
    )
    plot_compare_1d(
        wl_nm,
        e_in_scaled_pow,
        e_in_expected,
        ylabel="|E_in|²",
        title="Fabry |E_in|² scaling (4× I_new vs 4× expected)",
        filename="fabry_i_new_e_in_scaling.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        baseline_name="4× |E_in|² (base)",
        tmm_label="|E_in|² (4× I_new)",
    )
    assert np.allclose(e_in_scaled_pow, e_in_expected, rtol=1e-6)


def run_fdtd_input_combine_fabry_gin(sim, *, out_dir: Path) -> None:
    """lam_E_norm uses FDTD-equivalent passive T and fdtd_input_combine_power G_in."""
    sim_path = _fabry_sim_path()
    geom = parse_fabry_perot_geometry(sim_path)
    layout = parse_fdtd_layout(sim_path)
    wl_um, wl_nm = wl_grid_um(480.0, 520.0, 20)
    r_coeff = compute_fabry_reflection_ahead_of_z(
        layout.z_detector_in_um,
        geom,
        layout,
        wl_um,
        sim,
    )
    t_passive = compute_fabry_passive_transfer(
        geom,
        layout,
        wl_um,
        sim,
    )
    ratio = compute_fabry_lam_e_norm(
        geom, layout, wl_um, sim
    )
    expected = lam_e_norm_from_transfer(t_passive, r_coeff)
    max_abs = float(np.max(np.abs(ratio - expected)))
    log_before_visualize(
        "Fabry lam_E_norm — expected T/G_in vs computed helper",
        max_abs=f"{max_abs:.6g}",
        wl_range=f"{wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm",
    )
    plot_compare_1d(
        wl_nm,
        ratio,
        expected,
        ylabel="lam_E_norm",
        title="Fabry lam_E_norm: computed vs expected",
        filename="fabry_lam_e_norm.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        baseline_name="expected",
        tmm_label="computed",
    )
    assert np.allclose(ratio, expected, rtol=1e-6, atol=1e-12)
    assert np.allclose(ratio, t_passive / fdtd_input_combine_power(r_coeff), rtol=1e-6, atol=1e-12)


def run_fdtd_steady_field_ratio(sim, *, out_dir: Path) -> None:
    """Case2 |E_out|²/|E_in|² equals raw unit TMM field ratio (I_new cancels in norm)."""
    sim_path = _fabry_sim_path()
    geom = parse_fabry_perot_geometry(sim_path)
    layout = parse_fdtd_layout(sim_path)
    src_cfg = parse_fdtd_light_source_config(sim_path)
    wl_um, wl_nm = wl_grid_um(480.0, 520.0, 20)
    i_new = compute_fdtd_source_spectrum_i_new(wl_um, src_cfg)

    def build_layers(wl_val: float):
        return build_fabry_perot_fdtd_stack(
            geom, layout, wl_val, sim
        )

    e_in = compute_fdtd_steady_e_field_at_z(
        wl_um,
        src_cfg,
        layout.z_detector_in_um,
        build_layers,
        sim,
        i_new=i_new,
    )
    e_out = compute_fdtd_steady_e_field_at_z(
        wl_um,
        src_cfg,
        layout.z_detector_out_um,
        build_layers,
        sim,
        i_new=i_new,
    )
    t_ratio_case2 = np.abs(e_out) ** 2 / np.maximum(np.abs(e_in) ** 2, 1e-30)

    raw_in: list[complex] = []
    raw_out: list[complex] = []
    for w in wl_um:
        layers = build_fabry_perot_fdtd_stack(
            geom, layout, float(w), sim
        )
        raw_in.append(
            _cplx_from_py(
                sim.TMM_solver_passive_e_component_at_z_s(
                    layers,
                    float(w),
                    layout.z_detector_in_um,
                    complex(0.0, 0.0),
                    "s",
                )
            )
        )
        raw_out.append(
            _cplx_from_py(
                sim.TMM_solver_passive_e_component_at_z_s(
                    layers,
                    float(w),
                    layout.z_detector_out_um,
                    complex(0.0, 0.0),
                    "s",
                )
            )
        )
    ratio_raw = np.abs(raw_out) ** 2 / np.maximum(np.abs(raw_in) ** 2, 1e-30)
    max_abs = float(np.max(np.abs(t_ratio_case2 - ratio_raw)))
    log_before_visualize(
        "Fabry field ratio Case2 — FDTD steady vs passive TMM",
        max_abs=f"{max_abs:.6g}",
        wl_range=f"{wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm",
    )
    plot_compare_1d(
        wl_nm,
        t_ratio_case2,
        ratio_raw,
        ylabel="|E_out|²/|E_in|²",
        title="Fabry field ratio: FDTD steady vs passive TMM",
        filename="fabry_field_ratio_case2.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        baseline_name="passive TMM ratio",
        tmm_label="FDTD steady ratio",
    )
    assert np.allclose(t_ratio_case2, ratio_raw, rtol=1e-6, atol=1e-12)


def _run_check(name: str, fn, sim, out_dir: Path) -> tuple[str, str | None]:
    print(f"[RUN ] {name}", flush=True)
    try:
        fn(sim, out_dir=out_dir)
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}", flush=True)
        return "fail", str(exc)
    print(f"[ OK ] {name}", flush=True)
    return "ok", None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Oghma Bragg / Fabry-Pérot FDTD alignment checks."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_TMM / "output" / "test_oghma_bragg_fabry_alignment",
        help="PNG output directory.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = _load_simulation_module()
    bragg_ok = _bragg_sim_path().is_file()
    fabry_ok = _fabry_sim_path().is_file()

    print(
        f"Bragg project: {oghma_project_dir('bragg_grating', '01_hello_bragg_grating')} "
        f"({'found' if bragg_ok else 'missing'})"
    )
    print(
        f"Fabry project: {oghma_project_dir('fabry_perot', '01_hello_fabry_perot')} "
        f"({'found' if fabry_ok else 'missing'})"
    )
    print(f"Output dir: {out_dir.resolve()}")

    checks: list[tuple[str, Any]] = []
    for name in BRAGG_CHECKS:
        if not bragg_ok:
            print(f"[SKIP] {name}: Bragg FDTD project not found", flush=True)
            continue
        fn = globals()[f"run_{name}"]
        checks.append((name, fn))

    for name in FABRY_CHECKS:
        if not fabry_ok:
            print(f"[SKIP] {name}: Fabry-Pérot FDTD project not found", flush=True)
            continue
        fn = globals()[f"run_{name}"]
        checks.append((name, fn))

    if not checks:
        print("No checks to run (both projects missing).", flush=True)
        return 1

    failures: list[str] = []
    for name, fn in checks:
        status, err = _run_check(name, fn, sim, out_dir)
        if status == "fail":
            failures.append(f"{name}: {err}")

    if failures:
        print("FAILED:")
        for msg in failures:
            print(f"  {msg}")
        return 1

    print(f"Wrote plots to {out_dir.resolve()}")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
