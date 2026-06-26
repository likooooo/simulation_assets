"""Bragg grating passive TMM and FDTD alignment."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from oghma_core import (
    _compute_emission_case_bc_spectra,
    _compute_emission_coherent_fields_at_z,
    _compute_emission_transfer,
    _layers_from_builder,
    _make_oghma_optical_material,
    _predict_coupled_spectrum_coherent_p_bot,
    _tmm_layers_from_nk_thicknesses_um,
    lam_e_norm_from_transfer,
    load_oghma_material_nk,
    spectrum_reflection_coeff_unpolarized_s,
)
from oghma_fdtd import (
    FdtdLayout,
    _load_sim_dict,
    _oghma_length_m_to_um,
    _parse_world_segment0,
)
from oghma_fdtd_source import predict_fdtd_output_spectrum_incoherent

@dataclass(frozen=True)
class BraggGratingGeometry:
    """1D Bragg mirror from Oghma FDTD ``world`` replicated object."""

    n_periods: int
    period_um: float
    high_thickness_um: float
    low_thickness_um: float
    high_material: str
    low_nk: complex = 1.0 + 0.0j
    design_wavelength_um: float = 0.5

    @property
    def grating_thickness_um(self) -> float:
        return float(self.n_periods) * self.period_um



def parse_bragg_grating_geometry(sim_path: Path | str | dict[str, Any]) -> BraggGratingGeometry:
    """Parse Bragg mirror periods/thicknesses from ``sim.json`` ``world`` object."""
    seg = _parse_world_segment0(_load_sim_dict(sim_path))
    n_periods = int(seg.get("shape_nz", 1))
    period_um = _oghma_length_m_to_um(float(seg["dz_step"]))
    high_material = str(seg.get("optical_material", "generic/n/1.5"))

    box = seg.get("mesh", {}).get("box", {})
    if box and "dx" in box:
        d_h = _oghma_length_m_to_um(float(box["dx"]))
    else:
        n_h = float(np.real(load_oghma_material_nk(high_material, 0.5)))
        d_h = 0.5 / (4.0 * n_h)

    d_l = period_um - d_h
    if d_l <= 0:
        n_h = float(np.real(load_oghma_material_nk(high_material, 0.5)))
        d_h = period_um * n_h / (n_h + 1.0)
        d_l = period_um - d_h

    return BraggGratingGeometry(
        n_periods=n_periods,
        period_um=period_um,
        high_thickness_um=d_h,
        low_thickness_um=d_l,
        high_material=high_material,
    )


def _bragg_period_nk_thickness_um(
    geom: BraggGratingGeometry,
    lam_um: float,
    simulation_module: Any,
) -> tuple[list[complex], list[float]]:
    optical_high = _make_oghma_optical_material(simulation_module, geom.high_material)
    high_nk = complex(optical_high.nk_at_wavelength_um(float(lam_um)))
    nk_list: list[complex] = []
    thicknesses_um: list[float] = []
    for _ in range(geom.n_periods):
        nk_list.extend([high_nk, complex(geom.low_nk)])
        thicknesses_um.extend(
            [float(geom.high_thickness_um), float(geom.low_thickness_um)]
        )
    return nk_list, thicknesses_um


def build_bragg_grating_layers(
    geom: BraggGratingGeometry,
    lam_um: float,
    simulation_module: Any,
) -> list[Any]:
    """Passive Bragg mirror stack: ``air(∞) | [H|L]×N | air(∞)``."""
    period_nk, period_t = _bragg_period_nk_thickness_um(geom, lam_um, simulation_module)
    nk_list = [1.0 + 0.0j, *period_nk, 1.0 + 0.0j]
    thicknesses_um = [0.0, *period_t, 0.0]
    return _tmm_layers_from_nk_thicknesses_um(nk_list, thicknesses_um, simulation_module)


def build_bragg_fdtd_stack(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    lam_um: float,
    simulation_module: Any,
) -> list[Any]:
    """
    FDTD-equivalent passive stack: ``air(∞) | air(d_pre) | Bragg×N | air(d_post) | air(∞)``.

    ``z=0`` at bottom PML side; FDTD source at ``layout.z_source_um`` lies in the pre-grating air layer.
    """
    d_pre = float(layout.z_stack_start_um)
    z_grating_end = d_pre + geom.grating_thickness_um
    d_post = max(float(layout.z_detector_out_um) - z_grating_end, 0.0)

    period_nk, period_t = _bragg_period_nk_thickness_um(geom, lam_um, simulation_module)
    nk_list: list[complex] = [1.0 + 0.0j]
    thicknesses_um: list[float] = [0.0]
    if d_pre > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_pre)
    nk_list.extend(period_nk)
    thicknesses_um.extend(period_t)
    if d_post > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_post)
    nk_list.append(1.0 + 0.0j)
    thicknesses_um.append(0.0)
    return _tmm_layers_from_nk_thicknesses_um(nk_list, thicknesses_um, simulation_module)


def bragg_fdtd_stack_labels(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
) -> list[str]:
    """Finite-layer labels for :func:`build_bragg_fdtd_stack` (incident → exit)."""
    mat = geom.high_material.split("/")[-1]
    z_end = layout.z_stack_start_um + geom.grating_thickness_um
    d_post = max(layout.z_detector_out_um - z_end, 0.0)
    labels = ["pre(air)"]
    for _ in range(geom.n_periods):
        labels.extend([f"H(n={mat})", "L(air)"])
    if d_post > 0:
        labels.append("post(air)")
    return labels


def bragg_fdtd_stack_thicknesses_um(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
) -> list[float]:
    z_end = layout.z_stack_start_um + geom.grating_thickness_um
    d_post = max(layout.z_detector_out_um - z_end, 0.0)
    thicknesses = [0.0, float(layout.z_stack_start_um)]
    for _ in range(geom.n_periods):
        thicknesses.extend([geom.high_thickness_um, geom.low_thickness_um])
    if d_post > 0:
        thicknesses.append(d_post)
    thicknesses.append(0.0)
    return thicknesses


def bragg_fdtd_formula_from_geometry(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
) -> str:
    """Filmstack formula for :func:`build_bragg_fdtd_stack` bar-chart alignment."""
    z_end = float(layout.z_stack_start_um) + float(geom.grating_thickness_um)
    d_post = max(float(layout.z_detector_out_um) - z_end, 0.0)
    parts = ["air 0"]
    if float(layout.z_stack_start_um) > 0:
        parts.append(f"pre {float(layout.z_stack_start_um):.6f}")
    for _ in range(int(geom.n_periods)):
        parts.append(f"H {float(geom.high_thickness_um):.6f}")
        parts.append(f"L {float(geom.low_thickness_um):.6f}")
    if d_post > 0:
        parts.append(f"post {d_post:.6f}")
    parts.append("air 0")
    return " ".join(parts)


def bragg_fdtd_materials_db(geom: BraggGratingGeometry) -> dict[str, Any]:
    from oghma_core import _read_oghma_material

    air = _read_oghma_material("generic/n/1.0")
    high = _read_oghma_material(geom.high_material)
    return {"pre": air, "H": high, "L": air, "post": air}


def plot_bragg_fdtd_stack_section(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    simulation_module: Any,
    *,
    stack_title: str = "Bragg FDTD stack (passive)",
) -> None:
    from oghma_fdtd_alignment import plot_geometry_fdtd_stack_section

    plot_geometry_fdtd_stack_section(
        bragg_fdtd_formula_from_geometry(geom, layout),
        bragg_fdtd_materials_db(geom),
        layout,
        simulation_module,
        stack_thickness_um=float(geom.grating_thickness_um),
        stack_title=stack_title,
    )


def compute_bragg_passive_transfer(
    geom: BraggGratingGeometry,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """
    Normal-incidence passive ``T(λ)`` through ``air(∞) | [H|L]×N | air(∞)``.

    Spectral transfer for 1D FDTD ``S_out/S_in`` between detectors (Bragg mirror only).
    """

    def build_layers(wl: float) -> list[Any]:
        return build_bragg_grating_layers(
            geom, wl, simulation_module
        )

    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    layers = _layers_from_builder(build_layers, float(wl_arr[0]))
    angle_c = complex(float(incident_angle_rad), 0.0)
    t_list = simulation_module.TMM_solver_spectrum_passive_transfer_unpolarized_s(
        layers, list(wl_arr.astype(float)), angle_c
    )
    return np.asarray(t_list, dtype=float)


def build_bragg_passive_stack_ahead_of_z(
    z_um: float,
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    lam_um: float,
    simulation_module: Any,
) -> list[Any]:
    """
    Passive stack from ``z_um`` upward (+z): ``air(∞) | air(d_gap) | Bragg×N | air(d_post) | air(∞)``.

    ``d_gap = max(z_stack_start − z_um, 0)``.  Bottom semi-infinite air represents the
  medium below ``z_um``; used to compute ``r(λ)`` looking toward the grating.
    """
    d_gap = max(float(layout.z_stack_start_um) - float(z_um), 0.0)
    z_grating_end = float(layout.z_stack_start_um) + float(geom.grating_thickness_um)
    d_post = max(float(layout.z_detector_out_um) - z_grating_end, 0.0)

    period_nk, period_t = _bragg_period_nk_thickness_um(geom, lam_um, simulation_module)
    nk_list: list[complex] = [1.0 + 0.0j]
    thicknesses_um: list[float] = [0.0]
    if d_gap > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_gap)
    nk_list.extend(period_nk)
    thicknesses_um.extend(period_t)
    if d_post > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_post)
    nk_list.append(1.0 + 0.0j)
    thicknesses_um.append(0.0)
    return _tmm_layers_from_nk_thicknesses_um(nk_list, thicknesses_um, simulation_module)


def compute_bragg_reflection_ahead_of_z(
    z_um: float,
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """Complex amplitude ``r(λ)`` at ``z_um`` looking toward +z (TE/TM average)."""

    def build_layers(wl: float) -> list[Any]:
        return build_bragg_passive_stack_ahead_of_z(
            z_um,
            geom,
            layout,
            wl,
            simulation_module,
        )

    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    layers = _layers_from_builder(build_layers, float(wl_arr[0]))
    return spectrum_reflection_coeff_unpolarized_s(
        simulation_module,
        layers,
        wl_arr,
        incident_angle_rad=incident_angle_rad,
    )


def compute_bragg_lam_e_norm(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    z_g_in_nm: float | None = None,
) -> np.ndarray:
    """
    FDTD ``lam_E_norm`` / ``S_out/S_in`` model: ``T_passive(λ) / G_in(λ)``.

    ``G_in = |1 + r|²`` at ``z_g_in_nm`` (default ``layout.z_detector_in_um``).
    """
    z_g = float(z_g_in_nm if z_g_in_nm is not None else layout.z_detector_in_um)
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    r_coeff = compute_bragg_reflection_ahead_of_z(
        z_g,
        geom,
        layout,
        wl_arr,
        simulation_module,
    )
    t_passive = compute_bragg_passive_transfer(
        geom,
        wl_arr,
        simulation_module,
    )
    return lam_e_norm_from_transfer(t_passive, r_coeff)


def _bragg_build_fdtd_stack_at_wl(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    simulation_module: Any | None,
) -> Callable[[float], list[Any]]:

    def build_stack(wl: float) -> list[Any]:
        return build_bragg_fdtd_stack(
            geom,
            layout,
            wl,
            simulation_module,
        )

    return build_stack


def compute_bragg_emission_transfer(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
    z_g_in_nm: float | None = None,
) -> dict[str, np.ndarray]:
    """Emission TMM transfer at ``z_um``: ``P_top``, ``P_bot``, power ratios."""
    build_stack = _bragg_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)

    def reflection_at_z(z_g: float, wl_arr: np.ndarray) -> np.ndarray:
        return compute_bragg_reflection_ahead_of_z(
            z_g,
            geom,
            layout,
            wl_arr,
            simulation_module,
        )

    return _compute_emission_transfer(
        layout,
        wl_um,
        simulation_module,
        build_stack_at_wl=build_stack,
        reflection_at_z=reflection_at_z,
        z_um=z_um,
        u=u,
        orient=orient,
        z_g_in_nm=z_g_in_nm,
    )


def compute_bragg_emission_coherent_fields_at_z(
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
    spectrum: np.ndarray | None = None,
    n_u: int = 1,
    u_span: float | None = None,
) -> dict[str, np.ndarray]:
    """Coherent ``E_out_top/bot`` at ``z_um`` via emission source grid sweep."""
    build_stack = _bragg_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)
    return _compute_emission_coherent_fields_at_z(
        layout,
        wl_um,
        simulation_module,
        build_stack_at_wl=build_stack,
        z_um=z_um,
        u=u,
        orient=orient,
        spectrum=spectrum,
        n_u=n_u,
        u_span=u_span,
    )


def predict_bragg_coupled_spectrum_coherent_p_bot(
    simulation_module: Any,
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    e_bot: np.ndarray,
    *,
    u: float = 0.0,
) -> np.ndarray:
    """Coherent P_bot route: ``get_emission_flux(E_bot_sum(λ))`` per wavelength."""
    build_stack = _bragg_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)
    return _predict_coupled_spectrum_coherent_p_bot(
        simulation_module,
        wl_um,
        e_bot,
        build_stack_at_wl=build_stack,
        u=u,
    )


def compute_bragg_emission_case_bc_spectra(
    simulation_module: Any,
    geom: BraggGratingGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    s_in: np.ndarray,
    *,
    spectrum: np.ndarray | None = None,
    t_passive: np.ndarray | None = None,
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
) -> dict[str, np.ndarray]:
    """Case B: coherent ``flux(E_bot)``; Case C: ``T_passive × I_new``."""
    build_stack = _bragg_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)

    def compute_passive_transfer_fn(wl_arr: np.ndarray) -> np.ndarray:
        return compute_bragg_passive_transfer(
            geom,
            wl_arr,
            simulation_module,
        )

    return _compute_emission_case_bc_spectra(
        simulation_module,
        layout,
        wl_um,
        s_in,
        build_stack_at_wl=build_stack,
        compute_passive_transfer_fn=compute_passive_transfer_fn,
        predict_case_c_fn=predict_fdtd_output_spectrum_incoherent,
        spectrum=spectrum,
        t_passive=t_passive,
        z_um=z_um,
        u=u,
        orient=orient,
    )




