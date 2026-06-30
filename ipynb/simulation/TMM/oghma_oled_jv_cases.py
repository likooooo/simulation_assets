"""Shared OLED JV alignment cases for default / Gummel / Newton solvers."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oghma_core import compare_metrics, load_oghma_project
from oghma_oled_utils import (
    DEFAULT_OLED_JV_PROJECT,
    _optical_recombination_profile_dd,
    _photon_gen_profile,
    _read_oghma_rshunt_ohm_m2,
    build_oled_gummel_solver_from_project,
    build_oled_newton_solver_from_project,
    build_oled_solver_from_project,
    compute_coupled_eqe_spectrum,
    compute_oled_emission_outcoupling_maps_ito_al,
    compute_oled_jv_optical_coupled,
    compute_oled_jv_optical_coupled_gummel,
    compute_oled_jv_optical_coupled_newton,
    list_oghma_snapshots,
    load_oghma_emission_efficiency,
    load_oghma_escape_reference,
    load_oghma_jv_reference,
    load_oghma_snapshot,
    oghma_organic_y0_um,
    oghma_y_mesh_um,
)
from oghma_fdtd_alignment import plot_1d_compare
from oghma_runtime import tmm_output_dir
from oghma_pytest_helpers import (
    CompareCaseResult,
    PlotSeries,
    emit_plots,
    log10_series,
    print_case_result,
    secondary_to_primary_peak_ratio,
)


TURN_ON_V_MIN = 2.5
EQE_PLATEAU_V_MIN = 2.7

CASE_NAMES = (
    "load_jv_reference",
    "snapshots_available",
    "dd_equilibrium",
    "jv_turnon_region",
    "v_eqe_plateau",
    "v_luminance",
    "coupled_eqe_spectrum",
    "emission_efficiency_scaling",
)


@dataclass(frozen=True)
class OledJvSolverProfile:
    description: str
    out_subdir: str
    build_equilibrium_solver: Callable[[Any, Any], tuple[Any, np.ndarray]]
    equilibrium_residual: Callable[[Any], float]
    compute_jv_coupled: Callable[..., dict[str, Any]]
    coupled_eqe_mode: str  # "oghma_gen" | "dd_gen"


DEFAULT_PROFILE = OledJvSolverProfile(
    description="OLED JV / EQE / luminance checks vs Oghma 02_oled_jv baseline.",
    out_subdir="02_oled_jv",
    build_equilibrium_solver=build_oled_solver_from_project,
    equilibrium_residual=lambda solver: float(solver.last_gummel_residual()),
    compute_jv_coupled=compute_oled_jv_optical_coupled,
    coupled_eqe_mode="oghma_gen",
)

GUMMEL_PROFILE = OledJvSolverProfile(
    description="OLED Gummel DD JV / EQE / luminance checks vs Oghma 02_oled_jv baseline.",
    out_subdir="02_oled_gummel_jv",
    build_equilibrium_solver=build_oled_gummel_solver_from_project,
    equilibrium_residual=lambda solver: float(solver.last_gummel_residual()),
    compute_jv_coupled=compute_oled_jv_optical_coupled_gummel,
    coupled_eqe_mode="dd_gen",
)

NEWTON_PROFILE = OledJvSolverProfile(
    description="OLED Newton DD JV / EQE / luminance checks vs Oghma 02_oled_jv baseline.",
    out_subdir="02_oled_newton_jv",
    build_equilibrium_solver=build_oled_newton_solver_from_project,
    equilibrium_residual=lambda solver: float(solver.last_newton_residual()),
    compute_jv_coupled=compute_oled_jv_optical_coupled_newton,
    coupled_eqe_mode="dd_gen",
)


def run_load_jv_reference(jv_ref: dict[str, Any]) -> CompareCaseResult:
    ok = (
        "jv_voltage" in jv_ref
        and "jv_current_density" in jv_ref
        and len(jv_ref["jv_voltage"]) == 391
        and len(jv_ref["v_eqe_voltage"]) == 391
        and len(jv_ref["v_luminance_voltage"]) == 391
    )
    return CompareCaseResult(
        name="load_jv_reference",
        passed=ok,
        message=(
            f"jv={len(jv_ref.get('jv_voltage', []))}, "
            f"v_eqe={len(jv_ref.get('v_eqe_voltage', []))}, "
            f"v_lum={len(jv_ref.get('v_luminance_voltage', []))}"
        ),
    )


def run_snapshots_available(project_path: Path) -> CompareCaseResult:
    snaps = list_oghma_snapshots(project_path)
    count = len(snaps)
    return CompareCaseResult(
        name="snapshots_available",
        passed=count >= 300,
        message=f"{count} snapshots",
    )


def run_dd_equilibrium(
    sim: Any,
    project: Any,
    profile: OledJvSolverProfile,
) -> CompareCaseResult:
    solver, _ = profile.build_equilibrium_solver(project, sim)
    ok = solver.solve_equilibrium()
    residual = profile.equilibrium_residual(solver)
    current = abs(float(solver.current_density()))
    passed = ok and residual < 1e-5 and current < 1e-5
    return CompareCaseResult(
        name="dd_equilibrium",
        passed=passed,
        message=f"residual={residual:.2e}, |J|={current:.2e}",
    )


def run_jv_turnon_region(coupled_result: dict[str, Any], jv_ref: dict[str, Any]) -> CompareCaseResult:
    v = coupled_result["voltage"]
    j = coupled_result["current_density"]
    vb = jv_ref["jv_voltage"]
    jb = jv_ref["jv_current_density"]
    mask = (vb >= TURN_ON_V_MIN) & (vb <= 4.0)
    x = vb[mask]
    j_ours = np.interp(x, v, j)
    jb_m = jb[mask]
    stats = compare_metrics(log10_series(j_ours), log10_series(jb_m))
    passed = stats["corr"] >= 0.98
    return CompareCaseResult(
        name="jv_turnon_region",
        passed=passed,
        stats=stats,
        message=f"corr threshold 0.98 in [{TURN_ON_V_MIN}, 4.0] V",
        plots=[
            PlotSeries(
                x=x,
                ours=j_ours,
                baseline=jb_m,
                ylabel="log10 J (A/m²)",
                title=f"J–V turn-on region ({TURN_ON_V_MIN}–4.0 V)",
                log_domain=True,
                filename="jv_turnon_compare.png",
            )
        ],
    )


def run_v_eqe_plateau(coupled_result: dict[str, Any], jv_ref: dict[str, Any]) -> CompareCaseResult:
    v = coupled_result["voltage"]
    eqe = coupled_result["v_eqe"]
    vb = jv_ref["v_eqe_voltage"]
    eb = jv_ref["v_eqe"]
    mask = (vb >= EQE_PLATEAU_V_MIN) & (vb <= 4.0)
    x = vb[mask]
    e_ours = np.interp(x, v, eqe)
    eb_m = eb[mask]
    stats = compare_metrics(e_ours, eb_m)
    active = e_ours[(e_ours > 1e-3) & (e_ours < 50.0) & np.isfinite(e_ours)]
    passed = active.size > 0 and float(np.nanmax(active)) > 0.1
    return CompareCaseResult(
        name="v_eqe_plateau",
        passed=passed,
        stats=stats,
        message=f"physics EQE >0.1% somewhere in [{EQE_PLATEAU_V_MIN}, 4.0] V (finite, <50%)",
        plots=[
            PlotSeries(
                x=x,
                ours=e_ours,
                baseline=eb_m,
                ylabel="EQE (%)",
                title=f"V–EQE plateau ({EQE_PLATEAU_V_MIN}–4.0 V)",
                filename="v_eqe_plateau_compare.png",
            )
        ],
    )


def run_v_luminance(coupled_result: dict[str, Any], jv_ref: dict[str, Any]) -> CompareCaseResult:
    v = coupled_result["voltage"]
    lum = coupled_result["v_luminance"]
    vb = jv_ref["v_luminance_voltage"]
    lb = jv_ref["v_luminance"]
    mask = (vb >= EQE_PLATEAU_V_MIN) & (vb <= 4.0)
    x = vb[mask]
    l_ours = np.interp(x, v, lum)
    lb_m = lb[mask]
    stats = compare_metrics(log10_series(l_ours), log10_series(lb_m))
    passed = stats["corr"] >= 0.75
    return CompareCaseResult(
        name="v_luminance",
        passed=passed,
        stats=stats,
        message=f"log(L) corr threshold 0.75 in [{EQE_PLATEAU_V_MIN}, 4.0] V",
        plots=[
            PlotSeries(
                x=x,
                ours=l_ours,
                baseline=lb_m,
                ylabel="log10 L (cd/m²)",
                title=f"V–L ({EQE_PLATEAU_V_MIN}–4.0 V)",
                log_domain=True,
                filename="v_luminance_compare.png",
            )
        ],
    )


def run_coupled_eqe_spectrum(
    sim: Any,
    project: Any,
    eta_esc_maps: dict[str, Any],
    project_path: Path,
    profile: OledJvSolverProfile,
) -> CompareCaseResult:
    ref_escape = load_oghma_escape_reference(project)
    test_voltages = [2.6, 3.0, 3.5]
    wl_um = eta_esc_maps["wl_um"]
    escape_eta = eta_esc_maps["eta_oghma_style"]
    escape_y = eta_esc_maps["y_um"]
    emission_wl = ref_escape.get("emission_wl_um", wl_um)
    emission_prob = ref_escape.get("emission_prob", np.ones_like(emission_wl))

    dd_solver = None
    mesh_y = None
    g_shunt = None
    eta_pl = None
    if profile.coupled_eqe_mode == "dd_gen":
        dd_solver, mesh_y = profile.build_equilibrium_solver(project, sim)
        g_shunt = 1.0 / _read_oghma_rshunt_ohm_m2(project.project_dir)
        eta_pl = load_oghma_emission_efficiency(project)

    plots: list[PlotSeries] = []
    worst_corr = 1.0
    worst_dd_corr = 1.0
    worst_peak_delta = 0.0
    for v_target in test_voltages:
        snaps = list_oghma_snapshots(project_path)
        snap_meta = min(snaps, key=lambda s: abs(s["voltage"] - v_target))
        snap = load_oghma_snapshot(project_path, snap_meta["index"])
        if "eqe_wl_um" not in snap or "photon_gen_rate" not in snap:
            return CompareCaseResult(
                name="coupled_eqe_spectrum",
                passed=True,
                skipped=True,
                message="snapshot EQE/gen data missing",
            )
        coupled_oghma = compute_coupled_eqe_spectrum(
            wl_um=wl_um,
            escape_eta=escape_eta,
            escape_y_um=escape_y,
            photon_gen_y_um=snap["gen_y_um"] + oghma_organic_y0_um(project),
            photon_gen_rate=snap["photon_gen_rate"],
            emission_wl_um=emission_wl,
            emission_prob=emission_prob,
            ref_eqe_wl_um=snap["eqe_wl_um"],
            ref_eqe=snap["eqe"],
        )
        ours = coupled_oghma["eqe_norm"]
        base = coupled_oghma["ref_eqe_norm"]
        stats = compare_metrics(ours, base)
        worst_corr = min(worst_corr, stats["corr"])
        v_label = f"{snap_meta['voltage']:.2f}".replace(".", "p")
        plots.append(
            PlotSeries(
                x=wl_um,
                ours=ours,
                baseline=base,
                ylabel="norm EQE(λ)",
                title=f"EQE spectrum @ V={snap_meta['voltage']:.2f} V"
                + (" (Oghma gen)" if profile.coupled_eqe_mode == "dd_gen" else ""),
                xlabel="λ (μm)",
                filename=f"eqe_spectrum_V{v_label}.png",
            )
        )

        if profile.coupled_eqe_mode == "dd_gen":
            assert dd_solver is not None and mesh_y is not None and g_shunt is not None and eta_pl is not None
            dd_solver.solve_at_voltage(float(snap_meta["voltage"]))
            j_terminal = float(dd_solver.current_density())
            r_dd = _optical_recombination_profile_dd(
                j_terminal, float(snap_meta["voltage"]), project, mesh_y, g_shunt
            )
            gen_dd = _photon_gen_profile(r_dd, mesh_y, eta_pl)
            coupled_dd = compute_coupled_eqe_spectrum(
                wl_um=wl_um,
                escape_eta=escape_eta,
                escape_y_um=escape_y,
                photon_gen_y_um=mesh_y,
                photon_gen_rate=gen_dd,
                emission_wl_um=emission_wl,
                emission_prob=emission_prob,
                ref_eqe_wl_um=snap["eqe_wl_um"],
                ref_eqe=snap["eqe"],
            )
            dd_stats = compare_metrics(coupled_dd["eqe_norm"], base)
            worst_dd_corr = min(worst_dd_corr, dd_stats["corr"])
            ref_peak_ratio = secondary_to_primary_peak_ratio(wl_um, base)
            ours_peak_ratio = secondary_to_primary_peak_ratio(wl_um, ours)
            if np.isfinite(ref_peak_ratio) and ref_peak_ratio > 0.0:
                worst_peak_delta = max(worst_peak_delta, abs(ours_peak_ratio - ref_peak_ratio))
            plots.append(
                PlotSeries(
                    x=wl_um,
                    ours=coupled_dd["eqe_norm"],
                    baseline=base,
                    ylabel="norm EQE(λ)",
                    title=f"EQE spectrum @ V={snap_meta['voltage']:.2f} V (DD gen)",
                    xlabel="λ (μm)",
                    filename=f"eqe_spectrum_dd_V{v_label}.png",
                )
            )

    if profile.coupled_eqe_mode == "dd_gen":
        passed = worst_corr >= 0.85 and worst_dd_corr >= 0.85 and worst_peak_delta <= 0.30
        message = (
            f"Oghma-gen corr>={0.85}, DD-gen corr>={0.85}, |Δ secondary peak|<=0.30 "
            f"over {test_voltages} V"
        )
        stats = {"corr": worst_corr, "dd_corr": worst_dd_corr, "peak_delta": worst_peak_delta}
    else:
        passed = worst_corr >= 0.85
        message = f"min corr threshold 0.85 over {test_voltages} V"
        stats = {"corr": worst_corr}

    return CompareCaseResult(
        name="coupled_eqe_spectrum",
        passed=passed,
        stats=stats,
        message=message,
        plots=plots,
    )


def run_emission_efficiency_scaling(
    sim: Any,
    project: Any,
    eta_esc_maps: dict[str, Any],
    profile: OledJvSolverProfile,
) -> CompareCaseResult:
    base = profile.compute_jv_coupled(project, sim, eta_esc_maps=eta_esc_maps, eta_pl=0.25)
    scaled = profile.compute_jv_coupled(project, sim, eta_esc_maps=eta_esc_maps, eta_pl=1.0)
    mask = base["voltage"] >= EQE_PLATEAU_V_MIN
    x = base["voltage"][mask]
    j_corr = compare_metrics(base["current_density"][mask], scaled["current_density"][mask])
    eqe_ratio = np.median(scaled["v_eqe"][mask] / np.maximum(base["v_eqe"][mask], 1e-30))
    lum_ratio = np.median(scaled["v_luminance"][mask] / np.maximum(base["v_luminance"][mask], 1e-30))
    passed = (
        j_corr["corr"] >= 0.999
        and abs(eqe_ratio - 4.0) < 0.2
        and abs(lum_ratio - 4.0) < 0.2
    )
    eta_title = "η_pl=1.0 vs η_pl=0.25" if profile.coupled_eqe_mode == "dd_gen" else "scaled vs base"
    return CompareCaseResult(
        name="emission_efficiency_scaling",
        passed=passed,
        stats=j_corr,
        message=f"eqe_ratio={eqe_ratio:.3f}, lum_ratio={lum_ratio:.3f} (expect ~4.0)",
        plots=[
            PlotSeries(
                x=x,
                ours=scaled["v_eqe"][mask],
                baseline=base["v_eqe"][mask],
                ylabel="EQE (%)",
                title=f"η_pl scaling: V–EQE ({eta_title})",
                filename="eta_pl_scaling_eqe.png",
            ),
            PlotSeries(
                x=x,
                ours=scaled["v_luminance"][mask],
                baseline=base["v_luminance"][mask],
                ylabel="L (cd/m²)",
                title=f"η_pl scaling: V–L ({eta_title})",
                filename="eta_pl_scaling_luminance.png",
            ),
        ],
    )


def build_session(
    project_path: Path,
    simulation_module: Any,
    profile: OledJvSolverProfile,
) -> dict[str, Any]:
    project = load_oghma_project(project_path)
    jv_ref = load_oghma_jv_reference(project_path)
    ref_escape = load_oghma_escape_reference(project)
    coupled_result = profile.compute_jv_coupled(
        project, simulation_module, ref_escape=ref_escape
    )
    return {
        "sim": simulation_module,
        "project": project,
        "project_path": project_path,
        "jv_ref": jv_ref,
        "eta_esc_maps": coupled_result["eta_esc_maps"],
        "coupled_result": coupled_result,
    }


def run_case(name: str, session: dict[str, Any], profile: OledJvSolverProfile) -> CompareCaseResult:
    if name == "load_jv_reference":
        return run_load_jv_reference(session["jv_ref"])
    if name == "snapshots_available":
        return run_snapshots_available(session["project_path"])
    if name == "dd_equilibrium":
        return run_dd_equilibrium(session["sim"], session["project"], profile)
    if name == "jv_turnon_region":
        return run_jv_turnon_region(session["coupled_result"], session["jv_ref"])
    if name == "v_eqe_plateau":
        return run_v_eqe_plateau(session["coupled_result"], session["jv_ref"])
    if name == "v_luminance":
        return run_v_luminance(session["coupled_result"], session["jv_ref"])
    if name == "coupled_eqe_spectrum":
        return run_coupled_eqe_spectrum(
            session["sim"],
            session["project"],
            session["eta_esc_maps"],
            session["project_path"],
            profile,
        )
    if name == "emission_efficiency_scaling":
        return run_emission_efficiency_scaling(
            session["sim"], session["project"], session["eta_esc_maps"], profile
        )
    raise ValueError(f"unknown case: {name}")


def run_oled_jv_alignment(
    *,
    profile: OledJvSolverProfile,
    simulation_module: Any,
    runtime_label: str,
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument("--project", type=Path, default=DEFAULT_OLED_JV_PROJECT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=tmm_output_dir(profile.out_subdir),
        help="PNG output directory.",
    )
    parser.add_argument(
        "--case",
        choices=CASE_NAMES,
        action="append",
        dest="cases",
        help="Run only selected case(s); repeatable.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    case_names = args.cases or list(CASE_NAMES)
    print(f"RUNTIME={runtime_label}")
    print(f"project={args.project}")

    session = build_session(args.project, simulation_module, profile)
    failures: list[str] = []

    for name in case_names:
        result = run_case(name, session, profile)
        print_case_result(result)
        emit_plots(result, out_dir, plot_1d_compare=plot_1d_compare)
        if result.skipped:
            continue
        if not result.passed:
            failures.append(name)

    print(f"Wrote plots to {out_dir.resolve()}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All selected cases passed.")
    return 0


def main_for_profile(
    profile: OledJvSolverProfile,
    *,
    runtime_label: str,
    argv: list[str] | None = None,
    simulation_module: Any | None = None,
) -> int:
    if simulation_module is None:
        from oghma_runtime import bootstrap_tmm_session

        bootstrap_tmm_session()
        import simulation as simulation_module  # noqa: WPS433

    return run_oled_jv_alignment(
        profile=profile,
        simulation_module=simulation_module,
        runtime_label=runtime_label,
        argv=argv or [],
    )


_PROFILES = {
    "default": DEFAULT_PROFILE,
    "gummel": GUMMEL_PROFILE,
    "newton": NEWTON_PROFILE,
}


def main_jv_cli(argv: list[str] | None = None) -> int:
    """CLI entry for ``test_oghma_oled_jv.py`` (--profile default|gummel|newton)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default="default",
        help="OLED DD solver profile for JV alignment.",
    )
    args, rest = parser.parse_known_args(argv)
    from oghma_runtime import bootstrap_tmm_session

    _, runtime, _ = bootstrap_tmm_session()
    import simulation  # noqa: WPS433

    return main_for_profile(
        _PROFILES[args.profile],
        runtime_label=str(runtime),
        argv=rest,
        simulation_module=simulation,
    )


_JV_PLOT_SOLVER_CONFIG = {
    "default": (compute_oled_jv_optical_coupled, "02_oled_jv"),
    "gummel": (compute_oled_jv_optical_coupled_gummel, "02_oled_gummel_jv"),
    "newton": (compute_oled_jv_optical_coupled_newton, "02_oled_newton_jv"),
}


def run_oled_jv_plot_workflow(
    *,
    project_path: Path = DEFAULT_OLED_JV_PROJECT,
    solver: str = "default",
    out_dir: Path | None = None,
    simulation_module: Any,
) -> Path:
    """Compute JV/EQE/L and write comparison plots (shared by ``02_oled_example.py``)."""
    compute_jv, default_subdir = _JV_PLOT_SOLVER_CONFIG[solver]
    out_dir = out_dir or Path("./output") / default_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    project = load_oghma_project(project_path)
    jv_ref = load_oghma_jv_reference(project_path)
    ref_escape = load_oghma_escape_reference(project)
    result = compute_jv(project, simulation_module, ref_escape=ref_escape)

    plot_1d_compare(
        jv_ref["jv_voltage"],
        np.interp(jv_ref["jv_voltage"], result["voltage"], result["current_density"]),
        jv_ref["jv_current_density"],
        "J (A/m²)",
        "J–V",
        baseline_name="Oghma baseline",
        xlabel="V (V)",
        tmm_label="ours",
        out_path=out_dir / "jv_compare.png",
    )
    plot_1d_compare(
        jv_ref["v_eqe_voltage"],
        np.interp(jv_ref["v_eqe_voltage"], result["voltage"], result["v_eqe"]),
        jv_ref["v_eqe"],
        "EQE (%)",
        "V–EQE",
        baseline_name="Oghma baseline",
        xlabel="V (V)",
        tmm_label="ours",
        out_path=out_dir / "v_eqe_compare.png",
    )
    plot_1d_compare(
        jv_ref["v_luminance_voltage"],
        np.interp(jv_ref["v_luminance_voltage"], result["voltage"], result["v_luminance"]),
        jv_ref["v_luminance"],
        "L (cd/m²)",
        "V–L",
        baseline_name="Oghma baseline",
        xlabel="V (V)",
        tmm_label="ours",
        out_path=out_dir / "v_luminance_compare.png",
    )

    for v_target in (2.6, 3.0, 3.5):
        snap = min(list_oghma_snapshots(project_path), key=lambda s: abs(s["voltage"] - v_target))
        loaded = load_oghma_snapshot(project_path, snap["index"])
        idx = int(np.argmin(np.abs(result["voltage"] - snap["voltage"])))
        coupled = result["eqe_spectra"][idx]
        if "eqe_wl_um" not in loaded:
            continue
        ref_on = np.interp(
            coupled["wl_um"], loaded["eqe_wl_um"], loaded["eqe"], left=0.0, right=0.0
        )
        ours = coupled["eqe_norm"]
        ours_n = ours / max(float(np.max(ours)), 1e-30)
        ref_n = ref_on / max(float(np.max(ref_on)), 1e-30)
        v_label = f"{v_target:.2f}".replace(".", "p")
        plot_1d_compare(
            coupled["wl_um"],
            ours_n,
            ref_n,
            "norm EQE(λ)",
            f"EQE spectrum @ V={snap['voltage']:.2f} V",
            baseline_name="Oghma baseline",
            xlabel="λ (μm)",
            tmm_label="ours",
            out_path=out_dir / f"eqe_spectrum_V{v_label}.png",
        )

    return out_dir
