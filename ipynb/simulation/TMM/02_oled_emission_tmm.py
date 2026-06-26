#!/usr/bin/env python3
"""OLED emission TMM outcoupling alignment — ours vs Oghma 02_hello_oled_emission_tmm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — spectrum / EQE 1D plots
import numpy as np
import pandas as pd

from oghma_runtime import oghma_project_dir, prepare_simulation, simulation_repo_root

_RUNTIME = prepare_simulation()
_REPO = simulation_repo_root()
import simulation  # noqa: E402
from oghma_core import (  # noqa: E402
    compare_arrays,
    compare_metrics,
    eml_z_bounds_um,
    epitaxy_table,
    format_oghma_epitaxy_roles,
    load_oghma_optical_reference,
    load_oghma_project,
    plot_2d_compare,
    resolve_oghma_materials_root,
)
from filmstack_visualizer import build_tmm_layers, layers_from_formula, plot_filmstack  # noqa: E402
from oghma_fdtd_alignment import plot_1d_compare  # noqa: E402
from oghma_oled_utils import oled_emission_formula_from_project, oled_emission_materials_db  # noqa: E402
from oghma_oled_utils import (  # noqa: E402
    compute_coupled_eqe_spectrum,
    compute_oled_emission_outcoupling_maps_ito_al,
    diagnose_oled_stack,
    electrical_params_table,
    format_oghma_halfspaces_ito_al,
    ito_contact_thickness_um,
    list_oghma_outcoupling_snapshots,
    list_oghma_snapshots,
    load_oghma_emission_efficiency,
    load_oghma_escape_reference,
    load_oghma_jv_reference,
    load_oghma_outcoupling_snapshot,
    load_oghma_snapshot,
    oghma_organic_y0_um,
    oghma_y_mesh_um,
    oled_emission_stack_labels_ito_al,
    peak_y_um,
    sample_on_ref_grid,
)
from oghma_oled_utils import build_oghma_oled_emission_stack_ito_al  # noqa: E402

DEFAULT_OLED_PROJECT = oghma_project_dir("oled", "01_hello_oled", repo=_REPO)


def pass_fail(stats, rmse_thr, max_thr, peak_thr=None, min_corr=None) -> str:
    ok = stats["rmse"] < rmse_thr and stats["max_abs"] < max_thr
    if min_corr is not None and np.isfinite(stats.get("corr", np.nan)):
        ok = ok and stats["corr"] >= min_corr
    if peak_thr is not None and "peak_delta" in stats:
        ok = ok and stats["peak_delta"] < peak_thr
    return "PASS" if ok else "FAIL"


def report_metrics(
    alignment_report: list[dict],
    name: str,
    tmm: np.ndarray,
    baseline: np.ndarray,
    *,
    x=None,
    rmse_thr=0.02,
    max_thr=0.05,
    peak_thr=None,
    min_corr=None,
) -> dict:
    stats = compare_metrics(
        tmm, baseline, x=x, peak_axis=0 if x is not None and np.ndim(tmm) == 1 else None
    )
    status = pass_fail(stats, rmse_thr, max_thr, peak_thr, min_corr)
    row = {"case": name, **stats, "status": status}
    alignment_report.append(row)
    print(
        f"[{status}] {name}: RMSE={stats['rmse']:.4g}, "
        f"max|err|={stats['max_abs']:.4g}, corr={stats.get('corr', float('nan')):.4f}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_OLED_PROJECT)
    parser.add_argument("--out-dir", type=Path, default=Path("./output/02_oled_emission_tmm"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    project = load_oghma_project(args.project)
    ref = load_oghma_escape_reference(project)
    load_oghma_optical_reference(project)
    load_oghma_jv_reference(args.project)
    eta_f2f = load_oghma_emission_efficiency(project)

    wl_um = project.wl_um
    y_oghma_um = oghma_y_mesh_um(project)
    ref_wl = ref.get("heat_wl_um", wl_um)
    ref_y = ref.get("heat_y_um", y_oghma_um)
    ito_t = ito_contact_thickness_um(project)
    alignment_report: list[dict] = []
    bl = " (baseline)"

    print(f"RUNTIME={_RUNTIME}")
    print(f"oghma_database/materials: {resolve_oghma_materials_root()}")
    print(f"project: {args.project}")
    print(f"simmode: {project.simmode}, thickness: {project.total_thickness_um:.3f} μm")
    print(f"λ: {wl_um[0]:.3f}–{wl_um[-1]:.3f} μm ({len(wl_um)} pts), y: {len(y_oghma_um)} pts")
    print(f"η_f2f: {eta_f2f}")

    print(format_oghma_epitaxy_roles(project))
    print(format_oghma_halfspaces_ito_al(project))
    labels = oled_emission_stack_labels_ito_al(project)
    print("TMM emission labels:", " | ".join(labels))

    diag = diagnose_oled_stack(
        project,
        wl_um[0],
        simulation,
        stack_builder=build_oghma_oled_emission_stack_ito_al,
    )
    print(f"TMM emission ITO/Al stack @ {wl_um[0]:.3f} μm (ITO epitaxy={ito_t:.3f} μm):")
    print(pd.DataFrame(diag).to_string(index=False))

    eml = project.eml_layer()
    if eml:
        _, _, z_src = eml_z_bounds_um(project)
        print(
            f"EML {eml.name}: y=[{eml.y0_um:.3f}, {eml.y1_um:.3f}] μm, z_src={z_src:.3f} μm"
        )
        print(f"  pl_input_spectrum={eml.pl_input_spectrum}, η_f2f={eml.pl_efficiency_f2f}")

    formula = oled_emission_formula_from_project(project)
    db = oled_emission_materials_db(project)
    materials, thicknesses_um = layers_from_formula(formula, db, simulation_module=simulation)
    plot_filmstack(
        build_tmm_layers(materials, thicknesses_um, simulation_module=simulation),
        title="Filmstack (z=0 @ ITO epitaxy, +z → Al; ITO/Al semi-inf via depth=0)",
        entry_bookend_below_z0=True,
        show=True,
    )
    epitaxy_table(project)

    print("Building η_oghma_style from Oghma reference formula...")
    maps = compute_oled_emission_outcoupling_maps_ito_al(
        project,
        wl_um,
        y_oghma_um,
        simulation,
        ref_escape=ref,
    )

    ref_esc = ref["heat_eta"]
    tmm_esc = sample_on_ref_grid(maps["eta_oghma_style"], wl_um, y_oghma_um, ref_wl, ref_y)
    plot_2d_compare(
        ref_wl,
        ref_y,
        tmm_esc,
        ref_esc,
        "escape formula verify: (heat_η/ph)×row_norm(ph) baseline",
        row_norm=False,
        out_path=out_dir / "escape_oghma_style_2d.png",
    )
    report_metrics(
        alignment_report,
        "escape η_oghma_style (formula verify)",
        tmm_esc,
        ref_esc,
        rmse_thr=0.20,
        max_thr=0.85,
        min_corr=0.70,
    )

    if eml is not None:
        yj = int(np.argmin(np.abs(ref_y - (eml.y0_um + eml.y1_um) / 2)))
        plot_1d_compare(
            ref_wl,
            tmm_esc[:, yj],
            ref_esc[:, yj],
            "η",
            f"escape @ y≈{ref_y[yj]:.3f} μm",
            "escape",
            xlabel="λ (μm)",
            out_path=out_dir / f"escape_cross_y{ref_y[yj]:.3f}um.png",
        )
        report_metrics(
            alignment_report,
            f"escape @ y={ref_y[yj]:.3f} μm",
            tmm_esc[:, yj],
            ref_esc[:, yj],
            x=ref_wl,
            rmse_thr=0.30,
            max_thr=0.55,
            min_corr=0.70,
        )

    ref_lam_avg = ref.get("lam_avg_eta")
    ref_lam_y = ref.get("lam_avg_y_um", ref_y)
    eta_lam_avg_ours = tmm_esc.mean(axis=0)
    eta_lam_avg_ours_on_ref = np.interp(ref_lam_y, ref_y, eta_lam_avg_ours)
    if ref_lam_avg is not None:
        plot_1d_compare(
            ref_lam_y,
            eta_lam_avg_ours_on_ref,
            ref_lam_avg,
            "⟨η⟩_λ",
            "λ-averaged escape vs y",
            "⟨escape⟩_λ",
            xlabel="y (μm)",
            out_path=out_dir / "lam_avg_escape_vs_y.png",
        )
        peak_delta = abs(
            peak_y_um(eta_lam_avg_ours_on_ref, ref_lam_y) - peak_y_um(ref_lam_avg, ref_lam_y)
        )
        stats_lam = compare_metrics(eta_lam_avg_ours_on_ref, ref_lam_avg)
        stats_lam["peak_delta"] = peak_delta
        report_metrics(
            alignment_report,
            "λ-avg escape vs y",
            eta_lam_avg_ours_on_ref,
            ref_lam_avg,
            x=ref_lam_y,
            rmse_thr=0.25,
            max_thr=0.50,
            peak_thr=0.23,
            min_corr=0.25,
        )
        print(f"Peak y delta: {peak_delta:.3f} μm")

    s_wl = ref.get("emission_wl_um", wl_um)
    s_prob = ref.get("emission_prob", np.ones_like(s_wl))
    s_on_wl = np.interp(wl_um, s_wl, s_prob, left=0.0, right=0.0)
    eta_vs_wl_avg_y = np.interp(wl_um, ref_wl, tmm_esc.mean(axis=1))
    i_norm = s_on_wl * eta_vs_wl_avg_y
    ref_esc_avg = np.interp(wl_um, ref_wl, ref_esc.mean(axis=1))
    i_base = s_on_wl * ref_esc_avg
    i_norm = i_norm / max(i_norm.max(), 1e-30)
    i_base_norm = i_base / max(i_base.max(), 1e-30)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(wl_um, i_norm, "C0-", label="S·⟨η⟩_y (ours)")
    ax.plot(wl_um, i_base_norm, "C1--", label=f"S·⟨escape⟩_y{bl}")
    ax.set_xlabel("λ (μm)")
    ax.set_ylabel("norm intensity")
    ax.set_title("Spectrum-weighted outcoupling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "spectrum_weighted_outcoupling.png", dpi=150)
    plt.close(fig)

    snap_dir = args.project / "snapshots"
    if snap_dir.is_dir():
        snap_list = list_oghma_snapshots(args.project)
        s_wl = ref.get("emission_wl_um", wl_um)
        s_prob = ref.get("emission_prob", np.ones_like(s_wl))
        print(args.project)
        for entry in snap_list[-3:]:
            idx = entry["index"]
            snap = load_oghma_snapshot(args.project, idx)
            if "eqe_wl_um" not in snap or "photon_gen_rate" not in snap:
                continue
            v = snap.get("voltage", float("nan"))
            coupled = compute_coupled_eqe_spectrum(
                wl_um=ref_wl,
                escape_eta=ref_esc,
                escape_y_um=ref_y,
                photon_gen_y_um=snap["gen_y_um"] + oghma_organic_y0_um(project),
                photon_gen_rate=snap["photon_gen_rate"],
                emission_wl_um=s_wl,
                emission_prob=s_prob,
                ref_eqe_wl_um=snap["eqe_wl_um"],
                ref_eqe=snap["eqe"],
            )
            wl = coupled["wl_um"]
            ours_n = coupled["eqe_norm"]
            base_n = coupled.get("ref_eqe_norm")
            if base_n is None:
                continue
            m = compare_arrays(ours_n, base_n)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(wl, ours_n, label="coupled (ours η_esc baseline)")
            ax.plot(wl, base_n, "--", label=f"EQE spectrum{bl}")
            ax.set_xlabel("λ (μm)")
            ax.set_ylabel("norm EQE")
            ax.set_title(f"EQE @ V={v:.2f} V — corr={m['corr']:.3f}")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.savefig(out_dir / f"eqe_spectrum_snap{idx}.png", dpi=150)
            plt.close(fig)
            report_metrics(
                alignment_report,
                f"EQE spectrum snap {idx}",
                ours_n,
                base_n,
                rmse_thr=0.5,
                max_thr=1.0,
                min_corr=0.50,
            )
    else:
        print("snapshots/ not in project — skip coupled EQE spectrum.")

    for entry in list_oghma_outcoupling_snapshots(args.project):
        wl_snap = entry.get("wl_um")
        if wl_snap is None or not np.isfinite(wl_snap):
            continue
        idx = entry["index"]
        snap = load_oghma_outcoupling_snapshot(args.project, idx)
        if "escape_prob" not in snap:
            continue
        y_snap = snap["y_um"]

        esc_ours = sample_on_ref_grid(
            maps["eta_oghma_style"], wl_um, y_oghma_um, np.array([wl_snap]), y_snap
        )[0]
        esc_base = snap["escape_prob"]

        plot_1d_compare(
            y_snap,
            esc_ours,
            esc_base,
            "η",
            f"snapshot {idx} escape @ λ={wl_snap:.3f} μm",
            "escape",
            xlabel="y (μm)",
            out_path=out_dir / f"snap{idx}_escape_{wl_snap:.3f}um.png",
        )
        report_metrics(
            alignment_report,
            f"snap {idx} escape @ {wl_snap:.3f} μm",
            esc_ours,
            esc_base,
            x=y_snap,
            rmse_thr=0.20,
            max_thr=0.45,
            min_corr=0.40,
        )

    summary = pd.DataFrame(alignment_report)
    summary_path = out_dir / "alignment_report.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Pass rate: {(summary['status'] == 'PASS').sum()} / {len(summary)}")
    print(pd.DataFrame(electrical_params_table(project)).to_string(index=False))
    print(f"Wrote plots and {summary_path} to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
