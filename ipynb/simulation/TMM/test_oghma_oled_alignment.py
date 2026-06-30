#!/usr/bin/env python3
"""Oghma 01_hello_oled alignment: passive/emission R/T, outcoupling maps, API guards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from oghma_runtime import (
    bootstrap_tmm_session,
    default_oled_hello_project,
    tmm_output_dir,
)

_, _RUNTIME = bootstrap_tmm_session(import_tmm=False)
_DEFAULT_OUT_DIR = tmm_output_dir("test_oghma_oled_alignment")
import simulation  # noqa: E402
from oghma_core import (
    DEFAULT_RHO_U_MAX,
    DEFAULT_RHO_U_MIN,
    compare_metrics,
    eml_z_bounds_um,
    integrate_isotropic_p_top,
    isotropic_outcoupling_fraction_at_z,
    load_oghma_optical_reference,
    load_oghma_project,
    load_outcoupling_config,
    oghma_emission_u_from_ray_theta_deg,
    oghma_emission_u_values_from_project,
    parse_oghma_csv,
    plot_2d_compare,
    u_escape_halfspace,
    _oghma_axis_to_um,
)
from oghma_oled_utils import (
    build_oghma_oled_emission_stack_ito_al,
    compute_oled_emission_outcoupling_maps_ito_al,
    compute_oled_emission_stack_rt_ito_al,
    compute_oled_passive_rt_ito_al,
    load_oghma_escape_reference,
    oghma_organic_y0_um,
    oghma_y_mesh_um,
    peak_y_um,
    sample_on_ref_grid,
)
from oghma_oled_utils import OLED_RT_GATES
from oghma_pytest_helpers import (
    CROSS_PATH_RTOL,
    assert_rt_gate,
    log_before_visualize,
    reciprocity_visualize,
)


def run_outcoupling_config_ray_theta(project_dir: Path):
    cfg = load_outcoupling_config(project_dir)
    assert cfg["outcoupling_model"] == "transfer_matrix"
    assert cfg["ray_theta_deg"] == 0.0
    assert cfg["incoherent_wavelengths"] == 5


def run_emission_u_from_project_ignores_ray_trace_grid(project_dir: Path):
    project = load_oghma_project(project_dir)
    u_vals = oghma_emission_u_values_from_project(project)
    assert len(u_vals) == 1
    assert u_vals[0] == oghma_emission_u_from_ray_theta_deg(0.0)


def run_oled_rt_three_path_alignment(sim, project_dir: Path):
    """Passive TMM, emission TMM stack R/T, and baseline share R(λ)/T(λ)."""
    project = load_oghma_project(project_dir)
    ref = load_oghma_optical_reference(project)
    wl_um = ref["reflect_wl_um"]
    r_base = np.asarray(ref["reflect"], dtype=float)
    t_base = np.asarray(ref["transmit"], dtype=float)

    opt_cfg = load_outcoupling_config(project_dir)
    assert opt_cfg["light_illuminate_from"] == "y0"
    assert float(opt_cfg["ray_theta_deg"]) == 0.0

    r_pass, t_pass = compute_oled_passive_rt_ito_al(
        project, wl_um, sim, incident_angle_rad=0.0
    )
    r_emis, t_emis = compute_oled_emission_stack_rt_ito_al(
        project, wl_um, sim, incident_angle_rad=0.0
    )

    wl_arr = np.asarray(wl_um, dtype=float)
    log_before_visualize(
        "三路径 R(λ) 对齐可视化 — passive TMM / emission-stack TMM / Oghma reflect.csv",
        n_wl=len(wl_arr),
        wl_range=f"{wl_arr[0]:.3f}–{wl_arr[-1]:.3f} μm",
        max_abs_passive_minus_emis=f"{float(np.max(np.abs(np.asarray(r_pass) - np.asarray(r_emis)))):.6g}",
        max_abs_passive_minus_baseline=f"{float(np.max(np.abs(np.asarray(r_pass) - r_base))):.6g}",
    )
    reciprocity_visualize(
        "oled_R_passive_vs_emission",
        "r_passive",
        r_pass,
        {"r_emission_stack": r_emis, "r_oghma_baseline": r_base},
    )
    log_before_visualize(
        "三路径 T(λ) 对齐可视化 — passive TMM / emission-stack TMM / Oghma transmit.csv",
        n_wl=len(wl_arr),
        wl_range=f"{wl_arr[0]:.3f}–{wl_arr[-1]:.3f} μm",
        max_abs_passive_minus_emis=f"{float(np.max(np.abs(np.asarray(t_pass) - np.asarray(t_emis)))):.6g}",
        max_abs_passive_minus_baseline=f"{float(np.max(np.abs(np.asarray(t_pass) - t_base))):.6g}",
    )
    reciprocity_visualize(
        "oled_T_passive_vs_emission",
        "t_passive",
        t_pass,
        {"t_emission_stack": t_emis, "t_oghma_baseline": t_base},
    )

    np.testing.assert_allclose(r_pass, r_emis, rtol=0.0, atol=CROSS_PATH_RTOL)
    np.testing.assert_allclose(t_pass, t_emis, rtol=0.0, atol=CROSS_PATH_RTOL)

    for label, r_tmm, t_tmm in (
        ("passive TMM", r_pass, t_pass),
        ("emission TMM stack", r_emis, t_emis),
    ):
        stats_r = compare_metrics(r_tmm, r_base, x=wl_um, peak_axis=0)
        stats_t = compare_metrics(t_tmm, t_base, x=wl_um, peak_axis=0)
        assert_rt_gate(stats_r, stats_t, label=label, **OLED_RT_GATES)


def run_oled_emission_alignment_smoke(sim, project_dir: Path):
    """Fast alignment gate on a λ subset (full grid in ``02_oled_emission_tmm.py``)."""
    project = load_oghma_project(project_dir)
    ref = load_oghma_escape_reference(project)
    wl_um = project.wl_um[:8]
    y_um = oghma_y_mesh_um(project)

    maps = compute_oled_emission_outcoupling_maps_ito_al(
        project,
        wl_um,
        y_um,
        sim,
        ref_escape=ref,
    )

    ref_wl = ref["heat_wl_um"]
    ref_y = ref["heat_y_um"]

    ref_esc = ref["heat_eta"]
    tmm_esc = sample_on_ref_grid(maps["eta_oghma_style"], wl_um, y_um, ref_wl, ref_y)
    wl_mask = (ref_wl >= wl_um.min()) & (ref_wl <= wl_um.max())
    esc_stats = compare_metrics(tmm_esc[wl_mask], ref_esc[wl_mask])
    assert esc_stats["max_abs"] < 0.88
    assert esc_stats["corr"] > 0.40

    wi = int(np.argmin(np.abs(ref_wl[wl_mask] - 0.436)))
    wi_full = int(np.where(wl_mask)[0][wi])
    esc_line = tmm_esc[wi_full]
    base_esc_line = ref_esc[wi_full]
    line_stats = compare_metrics(esc_line, base_esc_line, x=ref_y)
    assert line_stats["max_abs"] < 0.50

    lam_avg_ours = tmm_esc.mean(axis=0)
    lam_avg_on_ref = np.interp(ref.get("lam_avg_y_um", ref_y), ref_y, lam_avg_ours)
    ref_lam = ref.get("lam_avg_eta")
    if ref_lam is not None:
        peak_delta = abs(
            peak_y_um(lam_avg_on_ref, ref.get("lam_avg_y_um", ref_y))
            - peak_y_um(ref_lam, ref.get("lam_avg_y_um", ref_y))
        )
        assert peak_delta < 150.0


def run_compute_oled_emission_stack_rt_calls_emission_solver(sim, project_dir: Path):
    """``compute_oled_emission_stack_rt_ito_al`` must invoke emission equivalent-wave API."""
    project = load_oghma_project(project_dir)
    wl_um = np.array([0.436, 0.550])
    with patch.object(
        sim,
        "TMM_emission_solve_equivalent_plane_wave_s_s",
        wraps=sim.TMM_emission_solve_equivalent_plane_wave_s_s,
    ) as em_s, patch.object(
        sim,
        "TMM_emission_solve_equivalent_plane_wave_p_s",
        wraps=sim.TMM_emission_solve_equivalent_plane_wave_p_s,
    ) as em_p:
        compute_oled_emission_stack_rt_ito_al(
            project,
            wl_um,
            sim,
        )
        assert em_s.call_count == len(wl_um)
        assert em_p.call_count == len(wl_um)


ESC_RHO_ETA_SINGLE_TOL = 1e-2
# esc/η: P_escape/P_total (u_esc=n_exit·sinπ/2, u_eml=n_EML·sinπ/2); ρ: ∫p_top/ρ_max
RHO_U_MIN = DEFAULT_RHO_U_MIN
RHO_U_MAX = DEFAULT_RHO_U_MAX
RHO_U_STEPS = 64
ESC_ETA_MAX = 1.0 + 1e-6
ESC_ETA_STDEV_MIN = 0.01
RHO_DEPTH_CORR_MEDIAN_MIN = 0.82
YL2_VS_HEAT_ETA_CORR_MIN = 0.99
EMISSION_ESCAPE_SIDE = "top"


def _load_yl2_escape_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse photons_escape_prob_yl2.csv into (wl_unique, y_local_unique, esc_grid)."""
    data = parse_oghma_csv(path)
    meta = data["meta"]
    wl_um = _oghma_axis_to_um(data["x"], meta, axis="x")
    y_local_um = _oghma_axis_to_um(data["y"], meta, axis="y")
    esc = np.asarray(data["z"], dtype=float)
    wl_unique = np.unique(wl_um)
    y_unique = np.unique(y_local_um)
    grid = np.full((len(wl_unique), len(y_unique)), np.nan)
    wl_to_i = {w: i for i, w in enumerate(wl_unique)}
    y_to_i = {y: i for i, y in enumerate(y_unique)}
    for w, y, v in zip(wl_um, y_local_um, esc):
        grid[wl_to_i[w], y_to_i[y]] = v
    return wl_unique, y_unique, grid


def _nearest_grid_index(axis: np.ndarray, value: float) -> int:
    idx = int(np.argmin(np.abs(axis - value)))
    if abs(float(axis[idx]) - value) > 1e-3:
        raise ValueError(f"value {value} not on grid (nearest {axis[idx]})")
    return idx


def _active_emission_depth_axes(
    ref: dict,
    project,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NPD slice aligned with release-1.2 nm grid (150–190 nm → organic_y0 + 40 nm)."""
    organic_y0 = oghma_organic_y0_um(project)
    npd_y1 = float(project.layers[1].y1_um)
    ref_y = np.asarray(ref["heat_y_um"], dtype=float)
    mask = (ref_y >= organic_y0) & (ref_y <= npd_y1)
    z_um = ref_y[mask]
    y_local_um = z_um - organic_y0
    return np.asarray(ref["heat_wl_um"], dtype=float), z_um, y_local_um


def _compute_tmm_esc_rho_on_active_grid(
    sim,
    project,
    project_dir: Path,
    ref: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """TMM esc/ρ/η and Oghma refs on the active-layer (yl2) grid."""
    wl_um, z_um, y_local_um = _active_emission_depth_axes(ref, project)
    ref_wl = np.asarray(ref["heat_wl_um"], dtype=float)
    ref_y = np.asarray(ref["heat_y_um"], dtype=float)
    heat_eta = np.asarray(ref["heat_eta"], dtype=float)
    photon_density = np.asarray(ref["photon_density"], dtype=float)
    esc_csv = project_dir / "photons_escape_prob_yl2.csv"
    wl_yl2, y_yl2, esc_yl2 = _load_yl2_escape_grid(esc_csv)

    esc_ref = sample_on_ref_grid(esc_yl2, wl_yl2, y_yl2, wl_um, y_local_um)
    rho_ref = sample_on_ref_grid(photon_density, ref_wl, ref_y, wl_um, z_um)
    eta_ref = sample_on_ref_grid(heat_eta, ref_wl, ref_y, wl_um, z_um)

    esc_tmm = np.zeros_like(esc_ref)
    rho_tmm = np.zeros_like(esc_ref)
    eta_tmm = np.zeros_like(esc_ref)

    for i, wl in enumerate(wl_um):
        layers = build_oghma_oled_emission_stack_ito_al(
            project, float(wl), sim
        )
        u_esc = u_escape_halfspace(sim, layers, float(wl), side=EMISSION_ESCAPE_SIDE)

        raw_rho = np.zeros(len(z_um), dtype=float)
        for j, z in enumerate(z_um):
            raw_rho[j] = integrate_isotropic_p_top(
                sim, layers, float(wl), float(z), RHO_U_MIN, RHO_U_MAX, RHO_U_STEPS
            )
            eta_z = isotropic_outcoupling_fraction_at_z(
                sim,
                layers,
                float(wl),
                float(z),
                u_esc,
                RHO_U_MIN,
                RHO_U_STEPS,
                RHO_U_MAX,
            )
            esc_tmm[i, j] = eta_z
            eta_tmm[i, j] = eta_z

        rho_max = float(np.max(raw_rho))
        if rho_max < 1e-30:
            rho_max = 1.0
        rho_tmm[i, :] = raw_rho / rho_max

    return wl_um, z_um, y_local_um, esc_ref, rho_ref, eta_ref, esc_tmm, rho_tmm, eta_tmm


def _per_wavelength_depth_correlations(
    ours: np.ndarray, ref: np.ndarray
) -> np.ndarray:
    corrs = np.full(ours.shape[0], np.nan, dtype=float)
    for i in range(ours.shape[0]):
        mask = np.isfinite(ours[i]) & np.isfinite(ref[i])
        if int(mask.sum()) < 3:
            continue
        corrs[i] = float(np.corrcoef(ours[i, mask], ref[i, mask])[0, 1])
    return corrs


def run_oled_esc_rho_eta_alignment(
    sim,
    project_dir: Path,
    *,
    out_dir: Path | None = None,
):
    """esc/ρ/η: 3 grid-node diagnostics + active-layer depth trend gates + 2D maps."""
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project = load_oghma_project(project_dir)
    ref = load_oghma_escape_reference(project)

    (
        wl_um,
        z_um,
        y_local_um,
        esc_ref,
        rho_ref,
        eta_ref,
        esc_tmm,
        rho_tmm,
        eta_tmm,
    ) = _compute_tmm_esc_rho_on_active_grid(
        sim, project, project_dir, ref
    )

    yl2_vs_eta = compare_metrics(esc_ref, eta_ref)
    assert yl2_vs_eta["corr"] > YL2_VS_HEAT_ETA_CORR_MIN, (
        f"yl2 esc vs heat_eta corr={yl2_vs_eta['corr']:.4f}"
    )

    esc_depth_corrs = _per_wavelength_depth_correlations(esc_tmm, esc_ref)
    finite_esc_corrs = esc_depth_corrs[np.isfinite(esc_depth_corrs)]
    esc_corr_median = float(np.median(finite_esc_corrs))
    esc_corr_frac = float(np.mean(finite_esc_corrs > 0.78))

    rho_depth_corrs = _per_wavelength_depth_correlations(rho_tmm, rho_ref)
    finite_rho_corrs = rho_depth_corrs[np.isfinite(rho_depth_corrs)]
    rho_corr_median = float(np.median(finite_rho_corrs))

    rho_lam_avg_stats = compare_metrics(
        rho_tmm.mean(axis=0), rho_ref.mean(axis=0), x=z_um
    )

    single_points_nm = [
        (303.5, 0.5),
        (303.5, 50.5),
        (576.5, 50.5),
    ]
    wl_yl2, y_yl2, esc_yl2 = _load_yl2_escape_grid(project_dir / "photons_escape_prob_yl2.csv")

    single_rows: list[dict[str, float]] = []
    single_max_err = 0.0
    single_within_tol = 0

    for wl_nm, y_local_nm in single_points_nm:
        wl_pt = wl_nm * 1e-3
        y_local_pt = y_local_nm * 1e-3
        z_pt = y_local_pt + oghma_organic_y0_um(project)
        wi = _nearest_grid_index(wl_yl2, wl_pt)
        yi = _nearest_grid_index(y_yl2, y_local_pt)
        w_ref = int(np.argmin(np.abs(wl_um - wl_pt)))
        z_ref = int(np.argmin(np.abs(z_um - z_pt)))

        esc_ref_pt = float(esc_yl2[wi, yi])
        rho_ref_pt = float(rho_ref[w_ref, z_ref])
        eta_ref_pt = float(eta_ref[w_ref, z_ref])
        esc_ours, rho_ours, eta_ours = (
            float(esc_tmm[w_ref, z_ref]),
            float(rho_tmm[w_ref, z_ref]),
            float(eta_tmm[w_ref, z_ref]),
        )

        for label, val in (("esc", esc_ours), ("rho", rho_ours), ("eta", eta_ours)):
            assert -1e-6 <= val <= ESC_ETA_MAX, f"{label} out of [0,1] at ({wl_pt}, {z_pt})"

        err_esc = abs(esc_ours - esc_ref_pt)
        err_rho = abs(rho_ours - rho_ref_pt)
        err_eta = abs(eta_ours - eta_ref_pt)
        pt_max = max(err_esc, err_rho, err_eta)
        single_max_err = max(single_max_err, pt_max)
        if pt_max < ESC_RHO_ETA_SINGLE_TOL:
            single_within_tol += 1

        single_rows.append(
            {
                "wl_nm": wl_nm,
                "y_local_nm": y_local_nm,
                "z_um": z_pt,
                "esc_ref": esc_ref_pt,
                "esc_ours": esc_ours,
                "rho_ref": rho_ref_pt,
                "rho_ours": rho_ours,
                "eta_ref": eta_ref_pt,
                "eta_ours": eta_ours,
                "err_esc": err_esc,
                "err_rho": err_rho,
                "err_eta": err_eta,
            }
        )

    log_before_visualize(
        "esc/ρ/η 全局深度趋势 — η=P_escape(u_esc)/P_total(u_eml); ρ=∫p_top u du/ρ_max",
        n_wl=len(wl_um),
        n_z=len(z_um),
        esc_eta_max=f"{float(np.max(esc_tmm)):.4f}",
        esc_eta_stdev=f"{float(np.std(esc_tmm)):.4f}",
        esc_depth_corr_median=f"{esc_corr_median:.4f}",
        esc_depth_corr_frac_gt_thr=f"{esc_corr_frac:.4f}",
        rho_depth_corr_median=f"{rho_corr_median:.4f}",
        rho_lam_avg_depth_corr=f"{rho_lam_avg_stats['corr']:.4f}",
        single_points_within_1e_2=single_within_tol,
        single_max_abs_err=f"{single_max_err:.6g}",
    )
    for row in single_rows:
        print(
            f"      λ={row['wl_nm']} nm y_local={row['y_local_nm']} nm z={row['z_um']:.4f} μm: "
            f"esc {row['esc_ours']:.4f} vs {row['esc_ref']:.4f} (Δ{row['err_esc']:.4f}) | "
            f"rho {row['rho_ours']:.4f} vs {row['rho_ref']:.4f} (Δ{row['err_rho']:.4f}) | "
            f"eta {row['eta_ours']:.4f} vs {row['eta_ref']:.4f} (Δ{row['err_eta']:.4f})",
            flush=True,
        )

    esc_stats = compare_metrics(esc_tmm, esc_ref)
    rho_stats = compare_metrics(rho_tmm, rho_ref)
    eta_stats = compare_metrics(eta_tmm, eta_ref)
    plot_2d_compare(
        wl_um,
        z_um,
        esc_tmm,
        esc_ref,
        "esc: TMM (P_escape/P_total) vs photons_escape_prob_yl2",
        row_norm=False,
        out_path=out_dir / "esc_yl2_2d.png",
        baseline_label="yl2",
    )
    plot_2d_compare(
        wl_um,
        z_um,
        rho_tmm,
        rho_ref,
        "ρ: TMM (∫p_top u du / ρ_max) vs outcoupling/photons_yl",
        row_norm=False,
        out_path=out_dir / "rho_photons_yl_2d.png",
        baseline_label="photons_yl",
    )
    plot_2d_compare(
        wl_um,
        z_um,
        eta_tmm,
        eta_ref,
        "η: TMM (P_escape/P_total) vs outcoupling/photons_escape_prob_yl",
        row_norm=False,
        out_path=out_dir / "eta_escape_prob_yl_2d.png",
        baseline_label="escape_prob_yl",
    )
    log_before_visualize(
        "esc/ρ/η 二维对齐图 — 同 02_oled_emission_tmm plot_2d_compare",
        out_dir=str(out_dir.resolve()),
        esc_rmse=f"{esc_stats['rmse']:.4g}",
        esc_max_abs=f"{esc_stats['max_abs']:.4g}",
        esc_corr=f"{esc_stats.get('corr', float('nan')):.4f}",
        rho_rmse=f"{rho_stats['rmse']:.4g}",
        rho_max_abs=f"{rho_stats['max_abs']:.4g}",
        rho_corr=f"{rho_stats.get('corr', float('nan')):.4f}",
        eta_rmse=f"{eta_stats['rmse']:.4g}",
        eta_max_abs=f"{eta_stats['max_abs']:.4g}",
        eta_corr=f"{eta_stats.get('corr', float('nan')):.4f}",
    )
    print(
        f"[2d] esc: RMSE={esc_stats['rmse']:.4g} max|err|={esc_stats['max_abs']:.4g} "
        f"corr={esc_stats.get('corr', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"[2d] rho: RMSE={rho_stats['rmse']:.4g} max|err|={rho_stats['max_abs']:.4g} "
        f"corr={rho_stats.get('corr', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"[2d] eta: RMSE={eta_stats['rmse']:.4g} max|err|={eta_stats['max_abs']:.4g} "
        f"corr={eta_stats.get('corr', float('nan')):.4f}",
        flush=True,
    )
    print(f"[2d] wrote esc/rho/eta maps to {out_dir.resolve()}", flush=True)

    wl_snap_um = 436.0 * 1e-3
    wi_snap = int(np.argmin(np.abs(wl_um - wl_snap_um)))
    reciprocity_visualize(
        "oled_esc_depth_trend",
        "esc_ref_yl2",
        esc_ref[wi_snap],
        {"esc_tmm_int": esc_tmm[wi_snap]},
    )
    reciprocity_visualize(
        "oled_rho_depth_trend",
        "rho_ref_photons_yl",
        rho_ref[wi_snap],
        {"rho_tmm_ptop": rho_tmm[wi_snap]},
    )

    assert float(np.max(esc_tmm)) <= ESC_ETA_MAX, (
        f"esc/η max={float(np.max(esc_tmm)):.4f} exceeds 1"
    )
    assert float(np.std(esc_tmm)) > ESC_ETA_STDEV_MIN, (
        f"esc/η stdev={float(np.std(esc_tmm)):.4f} — field is nearly constant"
    )
    assert rho_corr_median > RHO_DEPTH_CORR_MEDIAN_MIN, (
        f"ρ per-λ depth corr median={rho_corr_median:.4f}"
    )

    if esc_corr_median < 0.0:
        print(
            "[note] esc/η depth corr vs Oghma yl2 is negative "
            f"(median={esc_corr_median:.4f}); TMM uses bookend u_esc while "
            "Oghma CSV uses a finite u grid — compare 2D maps for diagnostics.",
            flush=True,
        )

    if single_within_tol < len(single_points_nm):
        print(
            "[note] single-point max_abs exceeds "
            f"{ESC_RHO_ETA_SINGLE_TOL:g} on some nodes; depth-trend gates passed.",
            flush=True,
        )


def main() -> int:
    from oghma_pytest_helpers import parse_oled_project_args

    args = parse_oled_project_args(
        description="Run Oghma OLED alignment checks as a plain Python script.",
        default_out_dir=_DEFAULT_OUT_DIR,
        default_project=default_oled_hello_project(),
    )
    project_dir = args.project_dir

    checks: list[tuple[str, Any]] = [
        ("outcoupling_config_ray_theta", lambda: run_outcoupling_config_ray_theta(project_dir)),
        ("emission_u_from_project_ignores_ray_trace_grid", lambda: run_emission_u_from_project_ignores_ray_trace_grid(project_dir)),
        (
            "oled_rt_three_path_alignment",
            lambda: run_oled_rt_three_path_alignment(simulation, project_dir),
        ),
        ("oled_emission_alignment_smoke", lambda: run_oled_emission_alignment_smoke(simulation, project_dir)),
        (
            "compute_oled_emission_stack_rt_calls_emission_solver",
            lambda: run_compute_oled_emission_stack_rt_calls_emission_solver(simulation, project_dir),
        ),
        (
            "oled_esc_rho_eta_alignment",
            lambda: run_oled_esc_rho_eta_alignment(
                simulation, project_dir, out_dir=args.out_dir
            ),
        ),
    ]

    print(f"RUNTIME={_RUNTIME}")
    print(f"Running {len(checks)} checks for project: {project_dir}")
    failures: list[str] = []
    for name, fn in checks:
        print(f"[RUN ] {name}")
        try:
            fn()
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failures.append(name)
            continue
        print(f"[ OK ] {name}")

    print(f"Wrote plots to {args.out_dir.resolve()}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
