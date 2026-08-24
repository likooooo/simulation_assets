"""Shared FDTD → TMM alignment: data loading and compare/report helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from py_core_plugins import viz_io
except ImportError:
    import viz_io

import matplotlib.pyplot as plt
import numpy as np

from oghma_core import (
    compare_metrics,
    compare_stop_band_metrics,
    peak_normalize_spectrum,
)
from oghma_fdtd import FdtdLayout, parse_fdtd_layout, parse_fdtd_source
from oghma_fdtd import load_oghma_fdtd_detector_reference
from oghma_fdtd_source import (
    FdtdLightSourceConfig,
    compute_fdtd_source_spectrum_i_new,
    load_oghma_light_spectrum,
    parse_fdtd_light_source_config,
)
from simulation_paths import resolve_artifacts_dir


@dataclass
class FdtdAlignmentBundle:
    """Single entry point for Oghma FDTD spectral alignment notebooks."""

    build: Path
    project_dir: Path
    sim_path: Path
    geom: Any
    layout: Any
    src_cfg: FdtdLightSourceConfig
    fdtd_src: Any
    ref: dict[str, Any]
    wl_um: np.ndarray
    s_in: np.ndarray
    s_out: np.ndarray
    i_new: np.ndarray
    alignment_report: list[dict[str, Any]] = field(default_factory=list)
    case_summary: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_fdtd_alignment_bundle(
    *project_parts: str,
    parse_geometry_fn: Callable[[Path], Any],
    normalize_i_new: bool = True,
) -> FdtdAlignmentBundle:
    """
    Load sim.json geometry/layout, detector spectra (``lam_E`` / ``lam_E_norm``), and ``I_new``.

    Spectral baselines come only from :func:`load_oghma_fdtd_detector_reference`.
    """
    from oghma_runtime import oghma_project_dir

    project_dir = oghma_project_dir(*project_parts)
    sim_path = project_dir / "sim.json"
    if not sim_path.is_file():
        raise FileNotFoundError(sim_path)
    geom = parse_geometry_fn(sim_path)
    layout = parse_fdtd_layout(sim_path)
    src_cfg = parse_fdtd_light_source_config(sim_path)
    fdtd_src = parse_fdtd_source(sim_path)
    ref = load_oghma_fdtd_detector_reference(project_dir)

    wl_um = np.asarray(ref["input_lam_e_wl_um"], dtype=float)
    s_in = np.asarray(ref["input_lam_e"], dtype=float)
    s_out = np.asarray(ref["output_lam_e"], dtype=float)

    i_new = compute_fdtd_source_spectrum_i_new(wl_um, src_cfg)
    if normalize_i_new:
        peak = max(float(np.max(i_new)), 1e-30)
        i_new = i_new / peak

    build = resolve_artifacts_dir()
    return FdtdAlignmentBundle(
        build=build,
        project_dir=project_dir,
        sim_path=sim_path,
        geom=geom,
        layout=layout,
        src_cfg=src_cfg,
        fdtd_src=fdtd_src,
        ref=ref,
        wl_um=wl_um,
        s_in=s_in,
        s_out=s_out,
        i_new=i_new,
    )


def s_in_times_intensity(bundle: FdtdAlignmentBundle) -> np.ndarray:
    """``S_IN × Intensity(λ)`` on ``bundle.wl_um`` for §2 envelope checks."""
    intensity = load_oghma_light_spectrum(
        bundle.src_cfg.light_spectrum,
        bundle.wl_um,
        multiplier=bundle.src_cfg.light_multiplier,
    )
    s_peak = max(float(np.max(bundle.s_in)), 1e-30)
    i_peak = max(float(np.max(intensity)), 1e-30)
    return bundle.s_in / s_peak * intensity / i_peak


def pass_fail(
    stats: dict[str, float],
    rmse_thr: float,
    max_thr: float,
    *,
    peak_thr: float | None = None,
    min_corr: float | None = None,
) -> str:
    ok = stats["rmse"] < rmse_thr and stats["max_abs"] < max_thr
    if min_corr is not None and np.isfinite(stats.get("corr", np.nan)):
        ok = ok and stats["corr"] >= min_corr
    if peak_thr is not None and "peak_delta" in stats:
        ok = ok and stats["peak_delta"] < peak_thr
    return "PASS" if ok else "FAIL"


def report_oled_metrics(
    alignment_report: list[dict[str, Any]],
    name: str,
    tmm: np.ndarray,
    baseline: np.ndarray,
    *,
    x: np.ndarray | None = None,
    rmse_thr: float = 0.02,
    max_thr: float = 0.05,
    peak_thr: float | None = None,
    min_corr: float | None = None,
) -> dict[str, float]:
    """OLED/optical notebooks: append compare row to *alignment_report* (not FdtdAlignmentBundle)."""
    peak_axis = 0 if x is not None and np.ndim(tmm) == 1 else None
    stats = compare_metrics(tmm, baseline, x=x, peak_axis=peak_axis)
    status = pass_fail(stats, rmse_thr, max_thr, peak_thr=peak_thr, min_corr=min_corr)
    row = {"case": name, **stats, "status": status}
    alignment_report.append(row)
    print(
        f"[{status}] {name}: RMSE={stats['rmse']:.4g}, "
        f"max|err|={stats['max_abs']:.4g}, corr={stats.get('corr', float('nan')):.4f}"
    )
    return stats


def report_metrics(
    bundle: FdtdAlignmentBundle,
    name: str,
    tmm: np.ndarray,
    baseline: np.ndarray,
    *,
    x: np.ndarray | None = None,
    rmse_thr: float = 0.15,
    max_thr: float = 0.3,
    peak_thr: float = 30.0,
    min_corr: float = 0.85,
    stop_thr: float = 0.03,
    enforce_pass: bool = True,
    use_stop_band: bool = True,
) -> dict[str, float]:
    stats = compare_metrics(tmm, baseline, x=x)
    if use_stop_band and x is not None and np.ndim(tmm) == 1:
        stats.update(compare_stop_band_metrics(tmm, baseline, x))
    status = pass_fail(stats, rmse_thr, max_thr, peak_thr=peak_thr, min_corr=min_corr)
    if enforce_pass and "stop_delta_um" in stats:
        status = "PASS" if status == "PASS" and stats["stop_delta_um"] < stop_thr else "FAIL"
    row = {"case": name, **stats, "status": status}
    bundle.alignment_report.append(row)
    print(
        f"[{status}] {name}: RMSE={stats['rmse']:.4f}, "
        f"max|err|={stats['max_abs']:.4f}, corr={stats.get('corr', float('nan')):.4f}"
    )
    if "stop_delta_um" in stats:
        print(
            f"  stop-band λ: ours={stats['stop_x_ours']:.4f} μm, "
            f"baseline={stats['stop_x_baseline']:.4f} μm, "
            f"Δλ={stats['stop_delta_um']:.4f} μm"
        )
    return stats


def report_envelope_metrics(
    bundle: FdtdAlignmentBundle,
    name: str,
    pred: np.ndarray,
    baseline: np.ndarray,
    wl_um: np.ndarray,
    *,
    min_corr: float = 0.85,
    enforce_pass: bool = False,
) -> dict[str, float]:
    pred_n = peak_normalize_spectrum(pred)
    base_n = peak_normalize_spectrum(baseline)
    stats = compare_metrics(pred_n, base_n, x=wl_um)
    stats.update(compare_stop_band_metrics(pred_n, base_n, wl_um))
    status = "PASS" if stats["corr"] >= min_corr else "FAIL"
    if not enforce_pass:
        status = f"{status} (envelope)"
    row = {"case": name, **stats, "status": status}
    bundle.alignment_report.append(row)
    print(f"[{status}] {name}: norm-corr={stats['corr']:.4f}, RMSE={stats['rmse']:.4f}")
    return stats


def plot_1d_compare(
    x: np.ndarray,
    tmm: np.ndarray,
    baseline: np.ndarray,
    ylabel: str,
    title: str,
    baseline_name: str = "FDTD",
    *,
    xlabel: str = "λ (μm)",
    tmm_label: str = "TMM",
    out_path: Path | str | None = None,
) -> None:
    err = tmm - baseline
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, tmm, label=tmm_label)
    axes[0].plot(x, baseline, "--", label=baseline_name)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    ymin, ymax = np.nanmin(baseline), np.nanmax(baseline)
    if ymin > 0 and ymax / max(ymin, 1e-30) > 50:
        axes[0].set_yscale("log")
    axes[1].plot(x, err, color="C3")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(f"Δ{ylabel}")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    else:
        if not viz_io.is_save_to_file():
            plt.show()
    plt.close(fig)


def _prepare_compare_series(
    pred: np.ndarray,
    baseline: np.ndarray,
    *,
    log_domain: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if log_domain:
        pred = np.log10(np.maximum(pred, 1e-30))
        baseline = np.log10(np.maximum(baseline, 1e-30))
    return pred, baseline


def _slice_compare(
    pred: np.ndarray,
    baseline: np.ndarray,
    wl_um: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask is None:
        return pred, baseline, wl_um
    return pred[mask], baseline[mask], wl_um[mask]


def evaluate_case(
    bundle: FdtdAlignmentBundle,
    name: str,
    pred: np.ndarray,
    baseline: np.ndarray,
    wl_um: np.ndarray | None = None,
    *,
    mask: np.ndarray | None = None,
    mask_label: str | None = None,
    log_domain: bool = False,
    ylabel: str = "value",
    title: str | None = None,
    tmm_label: str = "TMM",
    baseline_name: str = "FDTD",
    sanity_msg: str | None = None,
    plot: bool = False,
    use_stop_band: bool = True,
) -> dict[str, float]:
    """Unified compare entry: metrics + optional raw/norm plots + ``case_summary``."""
    if wl_um is None:
        wl_um = bundle.wl_um
    if title is None:
        title = f"{name} vs {baseline_name}"
    if sanity_msg:
        print(sanity_msg)

    pred_cmp, base_cmp = _prepare_compare_series(pred, baseline, log_domain=log_domain)
    domain_tag = "log10" if log_domain else "raw"

    stats = report_metrics(
        bundle,
        f"{name} ({domain_tag})",
        pred_cmp,
        base_cmp,
        x=wl_um,
        enforce_pass=False,
        use_stop_band=use_stop_band,
    )

    if mask is not None:
        label = mask_label or "region"
        p_m, b_m, x_m = _slice_compare(pred_cmp, base_cmp, wl_um, mask)
        report_metrics(
            bundle,
            f"{name} ({domain_tag}) [{label}]",
            p_m,
            b_m,
            x=wl_um[mask],
            enforce_pass=False,
            use_stop_band=use_stop_band,
        )

    stats_norm = report_envelope_metrics(bundle, f"{name} (norm)", pred, baseline, wl_um)
    if mask is not None:
        label = mask_label or "region"
        p_m, b_m, _ = _slice_compare(pred, baseline, wl_um, mask)
        report_envelope_metrics(
            bundle,
            f"{name} norm [{label}]",
            p_m,
            b_m,
            wl_um[mask],
        )

    if plot:
        plot_1d_compare(
            wl_um,
            pred_cmp,
            base_cmp,
            f"{ylabel} ({domain_tag})",
            f"{title} (raw)",
            baseline_name,
            tmm_label=tmm_label,
        )
        pred_n = peak_normalize_spectrum(pred)
        base_n = peak_normalize_spectrum(baseline)
        plot_1d_compare(
            wl_um,
            pred_n,
            base_n,
            f"{ylabel} (norm)",
            f"{title} (peak-normalized)",
            baseline_name,
            tmm_label=tmm_label,
        )

    bundle.case_summary[name] = {
        "corr": float(stats["corr"]),
        "rmse": float(stats["rmse"]),
        "norm_corr": float(stats_norm["corr"]),
        "rmse_norm": float(stats_norm["rmse"]),
        "log_domain": log_domain,
    }
    return stats


def plot_forward_source_diagnostic(
    bundle: FdtdAlignmentBundle,
    g_in: np.ndarray,
    s_pred_det0: np.ndarray,
    *,
    s_in_norm: np.ndarray | None = None,
) -> None:
    """§2 diagnostic: ``S_IN×Intensity``, ``I_new``, ``I_new×G_in``, and ``|1+r|²``."""
    if s_in_norm is None:
        s_in_norm = s_in_times_intensity(bundle)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(bundle.wl_um, s_in_norm, "b", label="S_IN (detector0) × Intensity")
    axes[0].plot(bundle.wl_um, bundle.i_new, "g--", label="I_new (forward)")
    axes[0].plot(bundle.wl_um, s_pred_det0, "r:", label="I_new × G_in")
    axes[0].set_ylabel("|E|²")
    axes[0].set_title("Forward source spectrum vs detector0")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(bundle.wl_um, g_in, "b", label="|1+r|² @ z_detector_in")
    axes[1].set_xlabel("λ (μm)")
    axes[1].set_ylabel("|1+r|²")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_case_summary_grid(
    bundle: FdtdAlignmentBundle,
    rows: list[dict[str, Any]],
) -> None:
    """
    Multi-row raw / peak-normalized summary.

    Each *row* dict: ``title``, ``pred``, ``baseline``, ``pred_label``, ``baseline_label``,
    optional ``ylabel`` (default ``|E|²``).
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        pred = np.asarray(row["pred"], dtype=float)
        base = np.asarray(row["baseline"], dtype=float)
        p_peak = max(float(np.max(pred)), 1e-30)
        b_peak = max(float(np.max(base)), 1e-30)
        ylabel = row.get("ylabel", "|E|²")
        pred_l = row.get("pred_label", "TMM")
        base_l = row.get("baseline_label", "baseline")
        title = row.get("title", f"row {i}")

        axes[i, 0].plot(bundle.wl_um, pred, label=pred_l)
        axes[i, 0].plot(bundle.wl_um, base, "--", label=base_l)
        axes[i, 0].set_ylabel(ylabel)
        axes[i, 0].set_title(f"{title} (raw)")
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(bundle.wl_um, pred / p_peak, label=pred_l)
        axes[i, 1].plot(bundle.wl_um, base / b_peak, "--", label=base_l)
        axes[i, 1].set_title(f"{title} (peak-normalized)")
        axes[i, 1].legend(fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("λ (μm)")
    axes[-1, 1].set_xlabel("λ (μm)")
    plt.tight_layout()
    plt.show()


LayoutMarker = tuple[float, str, str]


def plot_fdtd_layout_markers(
    layout: FdtdLayout,
    *,
    z_stack_end: float,
    extra_markers: Sequence[LayoutMarker] = (),
    title: str = "FDTD layout markers",
    x_margin_um: float = 0.2,
) -> None:
    """Vertical markers for source, finite stack bounds, detectors, and optional structure z."""
    markers: list[LayoutMarker] = [
        (float(layout.z_source_um), "source", "C1"),
        (float(layout.z_stack_start_um), "stack start", "C3"),
        *[(float(z), lab, c) for z, lab, c in extra_markers],
        (float(z_stack_end), "stack end", "C3"),
        (float(layout.z_detector_in_um), "detector0", "C2"),
        (float(layout.z_detector_out_um), "detector1", "C2"),
    ]

    fig, ax = plt.subplots(figsize=(10, 1.2))
    for z, lab, c in markers:
        ax.axvline(z, color=c, ls="--", alpha=0.8)
        ax.text(z, 0.5, lab, rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_xlim(-x_margin_um, float(layout.z_detector_out_um) + x_margin_um)
    ax.set_ylim(0, 1)
    ax.set_xlabel("z (µm)")
    ax.set_title(title)
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_geometry_fdtd_stack_section(
    formula: str,
    materials_db: dict[str, Any],
    layout: FdtdLayout,
    simulation_module: Any,
    *,
    stack_thickness_um: float,
    stack_title: str,
    extra_markers: Sequence[LayoutMarker] = (),
) -> None:
    """Shared Bragg/Fabry wrapper around :func:`plot_fdtd_stack_section`."""
    z_stack_end = float(layout.z_stack_start_um) + float(stack_thickness_um)
    plot_fdtd_stack_section(
        formula,
        materials_db,
        layout,
        simulation_module,
        z_stack_end=z_stack_end,
        extra_markers=extra_markers,
        stack_title=stack_title,
    )


def plot_fdtd_stack_section(
    formula: str,
    materials_db: dict[str, Any],
    layout: FdtdLayout,
    simulation_module: Any,
    *,
    z_stack_end: float,
    extra_markers: Sequence[LayoutMarker] = (),
    stack_title: str = "",
    markers_title: str = "FDTD layout markers",
    entry_bookend_below_z0: bool = False,
) -> None:
    """TMM stack bar chart plus FDTD source/detector/stack marker timeline."""
    from filmstack_visualizer import build_tmm_layers, layers_from_formula, plot_filmstack

    materials, thicknesses_um = layers_from_formula(
        formula, materials_db, simulation_module=simulation_module
    )
    layers = build_tmm_layers(materials, thicknesses_um, simulation_module=simulation_module)
    plot_filmstack(
        layers,
        title=stack_title,
        entry_bookend_below_z0=entry_bookend_below_z0,
        show=True,
    )
    plot_fdtd_layout_markers(
        layout,
        z_stack_end=z_stack_end,
        extra_markers=extra_markers,
        title=markers_title,
    )


def display_alignment_reports(bundle: FdtdAlignmentBundle) -> None:
    import pandas as pd
    from IPython.display import display

    _stop_um_labels = {
        "stop_x_ours": "stop_x_ours (μm)",
        "stop_x_baseline": "stop_x_baseline (μm)",
        "stop_delta_um": "stop_delta_um (μm)",
    }

    if bundle.case_summary:
        display(
            pd.DataFrame([{"case": k, **v} for k, v in bundle.case_summary.items()]).set_index(
                "case"
            )
        )
    df = pd.DataFrame(bundle.alignment_report)
    df = df.rename(columns={k: v for k, v in _stop_um_labels.items() if k in df.columns})
    display(df)
