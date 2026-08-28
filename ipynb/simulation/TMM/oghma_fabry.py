"""Fabry-Pérot FDTD → TMM geometry and RT coefficients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from oghma_core import (
    SUBSTRATE_DEPTH_UM,
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
from oghma_fdtd import FdtdLayout, _oghma_length_m_to_um
from oghma_fdtd_source import predict_fdtd_output_spectrum_incoherent

# --- Fabry-Pérot FDTD → TMM (04_fabry_perot) --------------------------------


@dataclass(frozen=True)
class FabryPerotGeometry:
    """1D Fabry-Pérot cavity from Oghma FDTD ``world`` replicated object."""

    mirror_thickness_um: float
    cavity_length_um: float
    mirror_material: str
    replicate_step_um: float
    cavity_nk: complex = 1.0 + 0.0j

    @property
    def total_film_thickness_um(self) -> float:
        return 2.0 * self.mirror_thickness_um + self.cavity_length_um

    @property
    def fundamental_resonance_um(self) -> float:
        n = float(np.real(self.cavity_nk))
        return 2.0 * n * self.cavity_length_um


def parse_fabry_perot_geometry(sim_path: Path | str | dict[str, Any]) -> FabryPerotGeometry:
    """Parse Fabry-Pérot mirror/cavity geometry from ``sim.json`` ``world`` object."""
    if isinstance(sim_path, dict):
        sim = sim_path
    else:
        sim = json.loads(Path(sim_path).read_text())
    world = sim.get("world", {})
    world_data = world.get("world_data", world)
    n_seg = int(world_data.get("segments", 0))
    if n_seg < 1:
        raise ValueError("sim.json has no world segments for Fabry-Pérot geometry")
    seg = world_data["segment0"]
    mirror_um = _oghma_length_m_to_um(float(seg["dz"]))
    step_um = _oghma_length_m_to_um(float(seg["dz_step"]))
    cavity_um = step_um - mirror_um
    if cavity_um <= 0:
        raise ValueError(f"invalid cavity length from dz_step - dz: {step_um} - {mirror_um}")
    return FabryPerotGeometry(
        mirror_thickness_um=mirror_um,
        cavity_length_um=cavity_um,
        mirror_material=str(seg.get("optical_material", "generic/n/3.0")),
        replicate_step_um=step_um,
    )


def _fabry_film_nk_thickness_um(
    geom: FabryPerotGeometry,
    lam_um: float,
    simulation_module: Any,
) -> tuple[list[complex], list[float]]:
    from coating_solver_test_util import material_nk_at_wl

    mirror_optical = _make_oghma_optical_material(simulation_module, geom.mirror_material)
    mirror_nk = material_nk_at_wl(mirror_optical, float(lam_um))
    return (
        [mirror_nk, complex(geom.cavity_nk), mirror_nk],
        [
            float(geom.mirror_thickness_um),
            float(geom.cavity_length_um),
            float(geom.mirror_thickness_um),
        ],
    )


def build_fabry_perot_stack(
    geom: FabryPerotGeometry,
    lam_um: float,
    simulation_module: Any,
    *,
    substrate_depth_nm: float = SUBSTRATE_DEPTH_UM,
) -> list[Any]:
    """Passive TMM stack: ``air(∞) | mirror | cavity | mirror | air(∞)``."""
    del substrate_depth_nm
    film_nk, film_t = _fabry_film_nk_thickness_um(geom, lam_um, simulation_module)
    nk_list = [1.0 + 0.0j, *film_nk, 1.0 + 0.0j]
    thicknesses_um = [0.0, *film_t, 0.0]
    return _tmm_layers_from_nk_thicknesses_um(nk_list, thicknesses_um, simulation_module)


def build_fabry_perot_fdtd_stack(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    lam_um: float,
    simulation_module: Any,
) -> list[Any]:
    """
    FDTD-equivalent passive stack: ``air(∞) | air(d_pre) | mirror | cavity | mirror | air(d_post) | air(∞)``.

    ``z=0`` at bottom PML; detector z positions map directly to TMM depth in this stack.
    """
    d_pre = float(layout.z_stack_start_um)
    z_stack_end = d_pre + geom.total_film_thickness_um
    d_post = max(float(layout.z_detector_out_um) - z_stack_end, 0.0)

    film_nk, film_t = _fabry_film_nk_thickness_um(geom, lam_um, simulation_module)
    nk_list: list[complex] = [1.0 + 0.0j]
    thicknesses_um: list[float] = [0.0]
    if d_pre > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_pre)
    nk_list.extend(film_nk)
    thicknesses_um.extend(film_t)
    if d_post > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_post)
    nk_list.append(1.0 + 0.0j)
    thicknesses_um.append(0.0)
    return _tmm_layers_from_nk_thicknesses_um(nk_list, thicknesses_um, simulation_module)


def build_fabry_fdtd_stack_ahead_of_z(
    z_um: float,
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    lam_um: float,
    simulation_module: Any,
) -> list[Any]:
    """
    Passive stack from ``z_um`` upward (+z): ``air(∞) | air(d_gap) | mirror | cavity | mirror | air(d_post) | air(∞)``.
    """
    d_gap = max(float(layout.z_stack_start_um) - float(z_um), 0.0)
    z_stack_end = float(layout.z_stack_start_um) + float(geom.total_film_thickness_um)
    d_post = max(float(layout.z_detector_out_um) - z_stack_end, 0.0)

    film_nk, film_t = _fabry_film_nk_thickness_um(geom, lam_um, simulation_module)
    nk_list: list[complex] = [1.0 + 0.0j]
    thicknesses_um: list[float] = [0.0]
    if d_gap > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_gap)
    nk_list.extend(film_nk)
    thicknesses_um.extend(film_t)
    if d_post > 0.0:
        nk_list.append(1.0 + 0.0j)
        thicknesses_um.append(d_post)
    nk_list.append(1.0 + 0.0j)
    thicknesses_um.append(0.0)
    return _tmm_layers_from_nk_thicknesses_um(nk_list, thicknesses_um, simulation_module)


def compute_fabry_reflection_ahead_of_z(
    z_um: float,
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """Complex amplitude ``r(λ)`` at ``z_um`` looking toward +z."""

    def build_layers(wl: float) -> list[Any]:
        return build_fabry_fdtd_stack_ahead_of_z(
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


def compute_fabry_passive_transfer(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """Normal-incidence passive ``T(λ)`` through the FDTD-equivalent Fabry–Pérot stack."""

    def build_layers(wl: float) -> list[Any]:
        return build_fabry_perot_fdtd_stack(
            geom,
            layout,
            wl,
            simulation_module,
        )

    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    layers = _layers_from_builder(build_layers, float(wl_arr[0]))
    angle_c = complex(float(incident_angle_rad), 0.0)
    t_list = simulation_module.TMM_solver_spectrum_passive_transfer_unpolarized_s(
        layers, list(wl_arr.astype(float)), angle_c
    )
    return np.asarray(t_list, dtype=float)


def compute_fabry_lam_e_norm(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    z_g_in_nm: float | None = None,
) -> np.ndarray:
    """FDTD ``lam_E_norm`` model: ``T_passive(λ) / G_in(λ)`` at ``z_detector_in``."""
    z_g = float(z_g_in_nm if z_g_in_nm is not None else layout.z_detector_in_um)
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    r_coeff = compute_fabry_reflection_ahead_of_z(
        z_g,
        geom,
        layout,
        wl_arr,
        simulation_module,
    )
    t_passive = compute_fabry_passive_transfer(
        geom,
        layout,
        wl_arr,
        simulation_module,
    )
    return lam_e_norm_from_transfer(t_passive, r_coeff)


def fabry_perot_fdtd_stack_labels(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
) -> list[str]:
    z_stack_end = layout.z_stack_start_um + geom.total_film_thickness_um
    d_post = max(layout.z_detector_out_um - z_stack_end, 0.0)
    mat = geom.mirror_material.split("/")[-1]
    labels = ["pre(air)"] if layout.z_stack_start_um > 0 else []
    labels.extend([f"mirror(n={mat})", "cavity(air)", f"mirror(n={mat})"])
    if d_post > 0:
        labels.append("post(air)")
    return labels


def fabry_perot_fdtd_stack_thicknesses_um(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
) -> list[float]:
    z_stack_end = layout.z_stack_start_um + geom.total_film_thickness_um
    d_post = max(layout.z_detector_out_um - z_stack_end, 0.0)
    thicknesses = [0.0]
    if layout.z_stack_start_um > 0:
        thicknesses.append(float(layout.z_stack_start_um))
    thicknesses.extend(
        [
            geom.mirror_thickness_um,
            geom.cavity_length_um,
            geom.mirror_thickness_um,
        ]
    )
    if d_post > 0:
        thicknesses.append(float(d_post))
    thicknesses.append(0.0)
    return thicknesses


def fabry_perot_stack_labels(geom: FabryPerotGeometry) -> list[str]:
    mat = geom.mirror_material.split("/")[-1]
    return [f"mirror(n={mat})", "cavity(air)", f"mirror(n={mat})"]


def fabry_perot_thicknesses_um(geom: FabryPerotGeometry) -> list[float]:
    return [
        0.0,
        geom.mirror_thickness_um,
        geom.cavity_length_um,
        geom.mirror_thickness_um,
        0.0,
    ]


def fabry_perot_fdtd_layout_extra_markers(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
) -> list[tuple[float, str, str]]:
    z_cavity_start = float(layout.z_stack_start_um) + float(geom.mirror_thickness_um)
    z_cavity_end = z_cavity_start + float(geom.cavity_length_um)
    return [
        (z_cavity_start, "cavity start", "C3"),
        (z_cavity_end, "cavity end", "C3"),
    ]


def fabry_perot_fdtd_formula_from_geometry(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
) -> str:
    """Filmstack formula for :func:`build_fabry_perot_fdtd_stack` bar-chart alignment."""
    z_stack_end = float(layout.z_stack_start_um) + float(geom.total_film_thickness_um)
    d_post = max(float(layout.z_detector_out_um) - z_stack_end, 0.0)
    parts = ["air 0"]
    if float(layout.z_stack_start_um) > 0:
        parts.append(f"pre {float(layout.z_stack_start_um):.6f}")
    parts.extend(
        [
            f"mirror {float(geom.mirror_thickness_um):.6f}",
            f"cavity {float(geom.cavity_length_um):.6f}",
            f"mirror {float(geom.mirror_thickness_um):.6f}",
        ]
    )
    if d_post > 0:
        parts.append(f"post {d_post:.6f}")
    parts.append("air 0")
    return " ".join(parts)


def fabry_perot_fdtd_materials_db(geom: FabryPerotGeometry) -> dict[str, Any]:
    from oghma_core import _read_oghma_material

    air = _read_oghma_material("generic/n/1.0")
    mirror = _read_oghma_material(geom.mirror_material)
    return {"pre": air, "mirror": mirror, "cavity": air, "post": air}


def plot_fabry_perot_fdtd_stack_section(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    simulation_module: Any,
    *,
    stack_title: str = "Fabry–Pérot stack (FDTD geometry)",
) -> None:
    from oghma_fdtd_alignment import plot_geometry_fdtd_stack_section

    plot_geometry_fdtd_stack_section(
        fabry_perot_fdtd_formula_from_geometry(geom, layout),
        fabry_perot_fdtd_materials_db(geom),
        layout,
        simulation_module,
        stack_thickness_um=float(geom.total_film_thickness_um),
        stack_title=stack_title,
        extra_markers=fabry_perot_fdtd_layout_extra_markers(geom, layout),
    )


def fabry_perot_interface_R_at_wl(
    geom: FabryPerotGeometry,
    wl_um: float | np.ndarray,
    simulation_module: Any,
) -> float | np.ndarray:
    """Power reflectivity R at air|mirror via TMM (``air|mirror slab|air``)."""
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    mirror_optical = _make_oghma_optical_material(simulation_module, geom.mirror_material)
    from coating_solver_test_util import material_nk_at_wl

    r_list: list[float] = []
    for wl in wl_arr:
        mirror_nk = material_nk_at_wl(mirror_optical, float(wl))
        layers = _tmm_layers_from_nk_thicknesses_um(
            [1.0 + 0.0j, mirror_nk, 1.0 + 0.0j],
            [0.0, float(geom.mirror_thickness_um), 0.0],
            simulation_module,
        )
        r_val, _ = simulation_module.TMM_solver_rt_power_unpolarized_s(
            layers, float(wl), complex(0.0, 0.0)
        )
        r_list.append(float(r_val))
    out = np.asarray(r_list, dtype=float)
    if np.ndim(wl_um) == 0:
        return float(out[0])
    return out


def _fabry_build_fdtd_stack_at_wl(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    simulation_module: Any | None,
) -> Callable[[float], list[Any]]:

    def build_stack(wl: float) -> list[Any]:
        return build_fabry_perot_fdtd_stack(
            geom,
            layout,
            wl,
            simulation_module,
        )

    return build_stack


def compute_fabry_emission_transfer(
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
    z_g_in_nm: float | None = None,
) -> dict[str, np.ndarray]:
    """Emission TMM transfer at ``z_um``: ``P_top``, ``P_bot``, and FDTD-style transfer ratios."""
    build_stack = _fabry_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)

    def reflection_at_z(z_g: float, wl_arr: np.ndarray) -> np.ndarray:
        return compute_fabry_reflection_ahead_of_z(
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


def compute_fabry_emission_coherent_fields_at_z(
    geom: FabryPerotGeometry,
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
    build_stack = _fabry_build_fdtd_stack_at_wl(
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


def predict_fabry_coupled_spectrum_coherent_p_bot(
    simulation_module: Any,
    geom: FabryPerotGeometry,
    layout: FdtdLayout,
    wl_um: float | np.ndarray,
    e_bot: np.ndarray,
    *,
    u: float = 0.0,
) -> np.ndarray:
    """Coherent P_bot route: ``get_emission_flux(E_bot_sum(λ))`` per wavelength."""
    build_stack = _fabry_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)
    return _predict_coupled_spectrum_coherent_p_bot(
        simulation_module,
        wl_um,
        e_bot,
        build_stack_at_wl=build_stack,
        u=u,
    )


def compute_fabry_emission_case_bc_spectra(
    simulation_module: Any,
    geom: FabryPerotGeometry,
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
    build_stack = _fabry_build_fdtd_stack_at_wl(
        geom, layout, simulation_module)

    def compute_passive_transfer_fn(wl_arr: np.ndarray) -> np.ndarray:
        return compute_fabry_passive_transfer(
            geom,
            layout,
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

