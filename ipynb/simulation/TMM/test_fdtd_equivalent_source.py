#!/usr/bin/env python3
"""Validate TMM + equivalent source spectrum I_new(λ) against Oghma FDTD detector spectra.

Case C (primary): S_pred = T_passive(λ) × I_new(λ)  vs  detector1/lam_E (S_out)
Case B (secondary): coherent flux(E_bot) with I_new     vs  detector1/lam_E (S_out)

Absolute amplitude may differ (FFT calibration); out_norm gates use independent
peak normalization (each spectrum / its own max) for dimensionless shape comparison.
All wavelengths are in µm internally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from oghma_bragg import compute_bragg_emission_case_bc_spectra, parse_bragg_grating_geometry
from oghma_core import compare_metrics, peak_normalize_spectrum
from oghma_fabry import compute_fabry_emission_case_bc_spectra, parse_fabry_perot_geometry
from oghma_fdtd_alignment import load_fdtd_alignment_bundle, plot_1d_compare
from oghma_pytest_helpers import log_before_visualize, plot_compare_1d
from oghma_runtime import bootstrap_tmm_session, oghma_project_dir

_TMM = Path(__file__).resolve().parent
_REPO, _RUNTIME, _ = bootstrap_tmm_session()

BRAGG_PROJECT = "oghma_projects/bragg_grating/01_hello_bragg_grating"
FABRY_PROJECT = "oghma_projects/fabry_perot/01_hello_fabry_perot"

BRAGG_CASE_C_CORR_MIN = 0.95
BRAGG_CASE_B_CORR_MIN = 0.85
FABRY_CASE_C_CORR_MIN = 0.95
FABRY_CASE_B_CORR_MIN = 0.90


def _load_simulation_module():
    import simulation  # noqa: WPS433

    return simulation


def _evaluate_case_bc(
    *,
    label: str,
    wl: np.ndarray,
    s_out: np.ndarray,
    s_pred_b: np.ndarray,
    s_pred_c: np.ndarray,
    out_dir: Path,
    case_b_mask: np.ndarray | None,
    case_c_corr_min: float,
    case_b_corr_min: float,
) -> None:
    stats_c_norm = compare_metrics(
        peak_normalize_spectrum(s_pred_c),
        peak_normalize_spectrum(s_out),
        x=wl,
    )
    log_before_visualize(
        f"{label} Case C — T_passive·I_new vs detector1/lam_E (full band)",
        norm_corr=f"{stats_c_norm['corr']:.4f}",
    )
    plot_compare_1d(
        wl,
        s_pred_c,
        s_out,
        ylabel="power",
        title=f"{label} Case C: T_passive·I_new vs detector1/lam_E",
        filename=f"{label.lower()}_case_c_vs_s_out.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        tmm_label="Case C (T×I_new)",
    )
    plot_compare_1d(
        wl,
        peak_normalize_spectrum(s_pred_c),
        peak_normalize_spectrum(s_out),
        ylabel="peak-normalized power",
        title=f"{label} Case C out_norm (peak-normalized vs detector1/lam_E)",
        filename=f"{label.lower()}_case_c_vs_s_out_norm.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        tmm_label="Case C (norm)",
        baseline_name="Oghma FDTD (norm)",
    )
    assert stats_c_norm["corr"] >= case_c_corr_min, (
        f"{label} Case C out_norm corr {stats_c_norm['corr']:.4f} < {case_c_corr_min}"
    )

    if case_b_mask is None:
        case_b_mask = np.ones_like(wl, dtype=bool)
    stats_b_norm = compare_metrics(
        peak_normalize_spectrum(s_pred_b[case_b_mask]),
        peak_normalize_spectrum(s_out[case_b_mask]),
        x=wl[case_b_mask],
    )
    log_before_visualize(
        f"{label} Case B — flux(E_bot) with I_new vs detector1/lam_E",
        norm_corr=f"{stats_b_norm['corr']:.4f}",
    )
    plot_compare_1d(
        wl[case_b_mask],
        s_pred_b[case_b_mask],
        s_out[case_b_mask],
        ylabel="power",
        title=f"{label} Case B: flux(E_bot) vs detector1/lam_E",
        filename=f"{label.lower()}_case_b_vs_s_out.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        tmm_label="Case B (flux)",
    )
    plot_compare_1d(
        wl[case_b_mask],
        peak_normalize_spectrum(s_pred_b[case_b_mask]),
        peak_normalize_spectrum(s_out[case_b_mask]),
        ylabel="peak-normalized power",
        title=f"{label} Case B out_norm (peak-normalized vs detector1/lam_E)",
        filename=f"{label.lower()}_case_b_vs_s_out_norm.png",
        out_dir=out_dir,
        plot_1d_compare=plot_1d_compare,
        tmm_label="Case B (norm)",
        baseline_name="Oghma FDTD (norm)",
    )
    assert stats_b_norm["corr"] >= case_b_corr_min, (
        f"{label} Case B out_norm corr {stats_b_norm['corr']:.4f} < {case_b_corr_min}"
    )


def run_bragg_equivalent_source(sim, *, out_dir: Path) -> None:
    bundle = load_fdtd_alignment_bundle(
        _REPO, BRAGG_PROJECT, parse_bragg_grating_geometry, normalize_i_new=False
    )
    out = compute_bragg_emission_case_bc_spectra(
        sim,
        bundle.geom,
        bundle.layout,
        bundle.wl_um,
        bundle.s_in,
        spectrum=bundle.i_new,
    )
    _evaluate_case_bc(
        label="Bragg",
        wl=bundle.wl_um,
        s_out=bundle.s_out,
        s_pred_b=out["s_pred_b"],
        s_pred_c=out["s_pred_c"],
        out_dir=out_dir,
        case_b_mask=(bundle.wl_um >= 0.43) & (bundle.wl_um <= 0.72),
        case_c_corr_min=BRAGG_CASE_C_CORR_MIN,
        case_b_corr_min=BRAGG_CASE_B_CORR_MIN,
    )


def run_fabry_equivalent_source(sim, *, out_dir: Path) -> None:
    bundle = load_fdtd_alignment_bundle(
        _REPO, FABRY_PROJECT, parse_fabry_perot_geometry, normalize_i_new=False
    )
    out = compute_fabry_emission_case_bc_spectra(
        sim,
        bundle.geom,
        bundle.layout,
        bundle.wl_um,
        bundle.s_in,
        spectrum=bundle.i_new,
    )
    _evaluate_case_bc(
        label="Fabry",
        wl=bundle.wl_um,
        s_out=bundle.s_out,
        s_pred_b=out["s_pred_b"],
        s_pred_c=out["s_pred_c"],
        out_dir=out_dir,
        case_b_mask=None,
        case_c_corr_min=FABRY_CASE_C_CORR_MIN,
        case_b_corr_min=FABRY_CASE_B_CORR_MIN,
    )


def _run_check(name: str, fn: Callable[..., None], sim, out_dir: Path) -> tuple[str, str | None]:
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
        description=(
            "Validate TMM + equivalent source I_new(λ) against Oghma FDTD detector spectra."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_TMM / "output" / "test_fdtd_equivalent_source",
        help="PNG output directory.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = _load_simulation_module()
    bragg_dir = oghma_project_dir("bragg_grating", "01_hello_bragg_grating", repo=_REPO)
    fabry_dir = oghma_project_dir("fabry_perot", "01_hello_fabry_perot", repo=_REPO)
    bragg_ok = (bragg_dir / "sim.json").is_file()
    fabry_ok = (fabry_dir / "sim.json").is_file()

    print(f"Bragg project: {bragg_dir} ({'found' if bragg_ok else 'missing'})")
    print(f"Fabry project: {fabry_dir} ({'found' if fabry_ok else 'missing'})")
    print(f"Output dir: {out_dir.resolve()}")

    checks: list[tuple[str, Callable[..., None]]] = []
    if bragg_ok:
        checks.append(("bragg_equivalent_source", run_bragg_equivalent_source))
    else:
        print("[SKIP] bragg_equivalent_source: Bragg FDTD project not found", flush=True)

    if fabry_ok:
        checks.append(("fabry_equivalent_source", run_fabry_equivalent_source))
    else:
        print("[SKIP] fabry_equivalent_source: Fabry-Pérot FDTD project not found", flush=True)

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
    print("All equivalent-source checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
