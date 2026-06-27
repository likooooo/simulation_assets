"""ITO/Al OLED stacks and re-export shim for Oghma TMM utilities."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from oghma_core import (
    OghmaProject,
    _cplx_from_py,
    _layers_from_builder,
    _list_indexed_snapshots,
    _load_indexed_snapshot,
    _make_oghma_optical_material,
    _oghma_axis_to_um,
    _read_oghma_material,
    _structure_pos_from_layers,
    _tmm_layers_from_project_formula,
    gpvdm_y_to_z_simulation_um,
    load_oghma_material_nk,
    load_optical_config,
    parse_oghma_csv,
)

# 01_hello_oled ITO/Al: UV tabulated nk @303 nm → max|ΔR|≈0.088 (``02_oled_tmm.ipynb``).
OLED_RT_GATES = dict(
    r_rmse=0.035,
    r_max=0.09,
    r_corr=0.99,
    t_rmse=0.02,
    t_max=0.05,
)


def diagnose_oled_stack(
    project: OghmaProject,
    lam_um: float,
    simulation_module: Any,
    *,
    stack_builder: Any = None,
) -> list[dict[str, Any]]:
    """Return per-layer nk, thickness, and TMM interface positions for inspection."""
    if stack_builder is None:
        stack_builder = build_oghma_oled_passive_stack_ito_al
    layers = stack_builder(project, float(lam_um), simulation_module)
    pos = _structure_pos_from_layers(layers)
    rows: list[dict[str, Any]] = []
    for i, layer in enumerate(layers):
        nk = layer.background_material.nk_at_wavelength_um(float(lam_um))
        z_start = float(pos[i]) if i < len(pos) else float("nan")
        z_end = float(pos[i + 1]) if i + 1 < len(pos) else float("nan")
        rows.append(
            {
                "index": i,
                "depth_um": float(layer.depth),
                "z_start_um": z_start,
                "z_end_um": z_end,
                "n": float(complex(nk).real),
                "k": float(complex(nk).imag),
                "label": (
                    project.layers[i - 1].name
                    if 0 < i <= len(project.layers)
                    else ("ITO(∞)" if i == 0 else "Al(∞)")
                ),
            }
        )
    return rows


def ito_contact_thickness_um(project: OghmaProject) -> float:
    """ITO anode contact thickness (μm); first epitaxy layer."""
    return float(project.layers[0].thickness_um)


def al_contact_thickness_um(project: OghmaProject) -> float:
    """Al cathode contact thickness (μm); last epitaxy layer."""
    return float(project.layers[-1].thickness_um)


def format_oghma_halfspaces_ito_al(project: OghmaProject) -> str:
    """Describe ITO/Al TMM exterior vs Oghma ABC boundaries."""
    opt = load_optical_config(project.project_dir)
    y0 = opt.get("boundary_y0", "?")
    y1 = opt.get("boundary_y1", "?")
    return (
        f"Oghma y0={y0} | epitaxy | y1={y1}; "
        f"TMM semi-inf ITO (layer 0, z→−∞) | epitaxy z∈[0,{project.total_thickness_um:g}] | "
        f"semi-inf Al (layer −1, z→+∞)"
    )


def _oled_layer_token(layer) -> str:
    return layer.name.split()[0].replace(":", "").replace("-", "")


def oled_emission_formula_from_project(project: OghmaProject) -> str:
    """Filmstack formula for ITO/Al bookend OLED stacks."""
    parts = ["ITO_pad 0"]
    for layer in project.layers:
        parts.append(f"{_oled_layer_token(layer)} {layer.thickness_um:.5f}")
    parts.append("Al_pad 0")
    return " ".join(parts)


def oled_emission_materials_db(project: OghmaProject) -> dict[str, Any]:
    """Materials map for :func:`oled_emission_formula_from_project`."""
    db = {
        "ITO_pad": _read_oghma_material(project.layers[0].optical_material),
        "Al_pad": _read_oghma_material(project.layers[-1].optical_material),
    }
    for layer in project.layers:
        db[_oled_layer_token(layer)] = _read_oghma_material(layer.optical_material)
    return db


def oled_emission_formula_air_boundary_from_project(project: OghmaProject) -> str:
    """Filmstack formula with air semi-infinite bookends."""
    parts = ["air 0"]
    for layer in project.layers:
        parts.append(f"{_oled_layer_token(layer)} {layer.thickness_um:.5f}")
    parts.append("air 0")
    return " ".join(parts)


def oled_emission_materials_db_air_boundary(project: OghmaProject) -> dict[str, Any]:
    """Materials map for :func:`oled_emission_formula_air_boundary_from_project`."""
    return {
        _oled_layer_token(layer): _read_oghma_material(layer.optical_material)
        for layer in project.layers
    }


def build_oghma_oled_passive_stack_ito_al(
    project: OghmaProject,
    lam_um: float,
    simulation_module: Any,
):
    """
    Passive TMM stack with ITO/Al half-planes (``03_hello_oled_tmm``).

    ``get_structure_pos`` places layer 0 at ``z=−∞`` and finite epitaxy from
    ``z=0``.  Bookends are **zero-thickness** ITO/Al layers (semi-infinite
    half-spaces), not finite pads::

        ITO(∞) | ITO → … → Al | Al(∞)
    """
    return _tmm_layers_from_project_formula(
        oled_emission_formula_from_project(project),
        oled_emission_materials_db(project),
        lam_um,
        simulation_module,
    )


def build_oghma_oled_emission_stack_ito_al(
    project: OghmaProject,
    lam_um: float,
    simulation_module: Any,
):
    """
    Emission TMM stack with ITO/Al half-planes (``02_hello_oled_emission_tmm``).

    Same layer order and ``z=0`` @ ITO (Oghma ``y=0``) as
    :func:`build_oghma_oled_passive_stack_ito_al`.  Dipole depth ``z = y``; ITO-side escape is ``p_top``.
    """
    return build_oghma_oled_passive_stack_ito_al(
        project, lam_um, simulation_module
    )

def build_oghma_oled_emission_stack_ito_al_air_boundary(
    project: OghmaProject,
    lam_um: float,
    simulation_module: Any,
):
    """
    Emission TMM stack like :func:`build_oghma_oled_emission_stack_ito_al` with air bookends.

    Same finite epitaxy as the ITO/Al stack, but semi-infinite half-spaces are air
    (``n=1``) instead of ITO/Al::

        air(∞) | ITO → … → Al | air(∞)
    """
    return _tmm_layers_from_project_formula(
        oled_emission_formula_air_boundary_from_project(project),
        oled_emission_materials_db_air_boundary(project),
        lam_um,
        simulation_module,
    )


def compute_oled_stack_rt_power_unpolarized(
    project: OghmaProject,
    wl_um: np.ndarray | float,
    simulation_module: Any,
    *,
    stack_builder: Any,
    incident_angle_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Passive plane-wave R(λ)/T(λ) on an ITO/Al OLED stack.

  ``stack_builder`` is typically :func:`build_oghma_oled_passive_stack_ito_al` or
    :func:`build_oghma_oled_emission_stack_ito_al`.  Normal incidence from the ITO
    entry (Oghma ``light_illuminate_from=y0``, ``ray_theta=0``) uses ``incident_angle_rad=0``.
    """
    def build_layers(wl: float):
        return stack_builder(
            project, wl, simulation_module
        )

    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    layers = _layers_from_builder(build_layers, float(wl_arr[0]))
    angle_c = complex(float(incident_angle_rad), 0.0)
    r_list, t_list = simulation_module.TMM_solver_spectrum_rt_power_unpolarized_s(
        layers, wl_arr.tolist(), angle_c
    )
    return np.asarray(r_list, dtype=float), np.asarray(t_list, dtype=float)


def compute_oled_emission_stack_rt(
    project: OghmaProject,
    wl_um: np.ndarray | float,
    simulation_module: Any,
    *,
    stack_builder: Any = build_oghma_oled_emission_stack_ito_al,
    incident_angle_rad: float = 0.0,
    z_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    R(λ)/T(λ) on emission filmstack via emission-TMM equivalent plane-wave solve.

    For each wavelength, this function calls emission APIs
    ``TMM_emission_solve_equivalent_plane_wave_s_s`` and
    ``TMM_emission_solve_equivalent_plane_wave_p_s`` to get boundary amplitudes
  ``E_out_top``/``E_out_bot`` that match passive ``r``/``t`` (see
    ``solve_equivalent_plane_wave_emission`` in reciprocity tests).

    On ITO/Al semi-infinite bookend stacks, passive illumination from ITO (Oghma
    ``light_illuminate_from=y0``) is equivalent to placing the source at the
    epitaxy exit depth ``z = total_thickness_um`` (emission vs passive direction).
    """

    def build_layers(wl: float):
        return stack_builder(
            project, wl, simulation_module
        )

    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    z_abs = (
        float(project.total_thickness_um)
        if z_um is None
        else float(z_um)
    )
    r_list: list[float] = []
    t_list: list[float] = []

    for wl in wl_arr:
        wl_f = float(wl)
        layers_wl = build_layers(wl_f)
        n1 = complex(layers_wl[0].background_material.nk_at_wavelength_um(wl_f))
        n2 = complex(layers_wl[-1].background_material.nk_at_wavelength_um(wl_f))
        u = n1 * np.sin(float(incident_angle_rad))
        u_c = complex(float(np.real(u)), float(np.imag(u)))

        res_s = simulation_module.TMM_emission_solve_equivalent_plane_wave_s_s(
            layers_wl, wl_f, u_c, z_abs
        )
        res_p = simulation_module.TMM_emission_solve_equivalent_plane_wave_p_s(
            layers_wl, wl_f, u_c, z_abs
        )
        r_s = _cplx_from_py(res_s.E_out_top)
        t_s = _cplx_from_py(res_s.E_out_bot)
        r_p = _cplx_from_py(res_p.E_out_top)
        t_p = _cplx_from_py(res_p.E_out_bot)

        dirs = simulation_module.TMM_propagate_direction_from_u_s(layers_wl, u_c, wl_f)
        th1 = _cplx_from_py(dirs[0])
        th2 = _cplx_from_py(dirs[-1])

        def _rt_power_from_coeff(r_c: complex, t_c: complex, pol: str) -> tuple[float, float]:
            inv_t = 1.0 / t_c
            m = [[inv_t, r_c * inv_t], [r_c * inv_t, inv_t]]
            fn = (
                simulation_module.TMM_get_r_t_power_from_tmm_s
                if pol == "s"
                else simulation_module.TMM_get_r_t_power_from_tmm_p
            )
            rt = list(fn(m, n1, th1, n2, th2))
            return float(rt[0]), float(rt[1])

        r_s_pow, t_s_pow = _rt_power_from_coeff(r_s, t_s, "s")
        r_p_pow, t_p_pow = _rt_power_from_coeff(r_p, t_p, "p")
        r_list.append(0.5 * (r_s_pow + r_p_pow))
        t_list.append(0.5 * (t_s_pow + t_p_pow))

    return np.asarray(r_list, dtype=float), np.asarray(t_list, dtype=float)


def compute_oled_emission_stack_rt_ito_al(
    project: OghmaProject,
    wl_um: np.ndarray | float,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
    z_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """R(λ)/T(λ) on ITO/Al emission filmstack (see :func:`compute_oled_emission_stack_rt`)."""
    return compute_oled_emission_stack_rt(
        project,
        wl_um,
        simulation_module,
        stack_builder=build_oghma_oled_emission_stack_ito_al,
        incident_angle_rad=incident_angle_rad,
        z_um=z_um,
    )


def compute_oled_passive_rt_ito_al(
    project: OghmaProject,
    wl_um: np.ndarray | float,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """R(λ)/T(λ) for ``03_hello_oled_tmm`` passive stack (ITO-side illumination)."""
    return compute_oled_stack_rt_power_unpolarized(
        project,
        wl_um,
        simulation_module,
        stack_builder=build_oghma_oled_passive_stack_ito_al,
        incident_angle_rad=incident_angle_rad,
    )


def compute_oled_passive_field_intensity_profile_ito_al(
    project: OghmaProject,
    wl_um: float,
    y_um: np.ndarray,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """Unpolarized |E|²(y) for ITO/Al passive y0; ``y_um`` is gpvdm/Oghma y (= TMM z)."""
    layers = build_oghma_oled_passive_stack_ito_al(
        project, float(wl_um), simulation_module,
    )
    z_um = gpvdm_y_to_z_simulation_um(
        np.asarray(y_um, dtype=float), total_thickness_um=project.total_thickness_um
    )
    return np.asarray([
        float(
            simulation_module.TMM_solver_field_intensity_unpolarized_at_z_s(
                layers,
                float(wl_um),
                float(z),
                complex(float(incident_angle_rad), 0.0),
            )
        )
        for z in z_um
    ])


def compute_oled_passive_absorption_profile_ito_al(
    project: OghmaProject,
    wl_um: float,
    y_um: np.ndarray,
    simulation_module: Any,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """Layer absorption density at one λ (ITO/Al passive y0, TE/TM average)."""
    layers = build_oghma_oled_passive_stack_ito_al(
        project, float(wl_um), simulation_module,
    )
    z_um = gpvdm_y_to_z_simulation_um(
        np.asarray(y_um, dtype=float), total_thickness_um=project.total_thickness_um
    )
    return np.asarray([
        float(
            simulation_module.TMM_solver_absorption_unpolarized_at_z_s(
                layers,
                float(wl_um),
                float(z),
                complex(float(incident_angle_rad), 0.0),
            )
        )
        for z in z_um
    ])


# ---------------------------------------------------------------------------
# OLED JV / EQE / luminance (02_oled_jv)
# ---------------------------------------------------------------------------

def _default_oled_jv_project() -> Path:
    import os

    db = os.environ.get("SIMULATION_DATABASE_DIR", "").strip()
    if db:
        return Path(db).resolve().parent / "oghma_projects" / "oled" / "02_oled_jv"
    return (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "oghma_projects"
        / "oled"
        / "02_oled_jv"
    )


DEFAULT_OLED_JV_PROJECT = _default_oled_jv_project()

# --- Physical constants (OLED JV / optical coupling) ---
ELEMENTARY_CHARGE_C = 1.602176634e-19  # elementary charge (C); J → electron flux
PLANCK_CONSTANT_J_S = 6.62607015e-34  # Planck constant (J·s)
SPEED_OF_LIGHT_M_PER_S = 299792458.0  # vacuum speed of light (m/s)
PHOTOPIC_PEAK_LUMINOUS_EFFICACY_lm_per_W = 683.0  # CIE 555 nm peak (lm/W)
MIN_NATIVE_RECOMB_PROFILE_SCALE_RATIO = 0.02  # min ∫R_native / (J/q) to trust DD R(y)
MAX_NATIVE_RECOMB_PROFILE_SCALE_RATIO = 50.0  # max ∫R_native / (J/q) to trust DD R(y)

# CIE 1931 photopic luminosity function V(λ), 5 nm grid (relative, peak ≈ 1 at 555 nm).
_CIE_1931_V_WL_NM = np.array(
    [
        380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450,
        455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525,
        530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600,
        605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
        680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750,
        755, 760, 765, 770, 775, 780,
    ],
    dtype=float,
)
_CIE_1931_V_VALUES = np.array(
    [
        0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073,
        0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739, 0.0910, 0.1126,
        0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932,
        0.8620, 0.9149, 0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
        0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412, 0.3810, 0.3210,
        0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232,
        0.0170, 0.0119, 0.0082, 0.0057, 0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007,
        0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000,
        0.0000,
    ],
    dtype=float,
)


def _read_oghma_rshunt_ohm_m2(project_dir: Path | str) -> float:
    sim = json.loads(Path(project_dir).joinpath("sim.json").read_text())
    return float(sim.get("parasitic", {}).get("Rshunt", 1.2))


def _read_jv_sweep_config(project_dir: Path | str) -> tuple[float, float, float]:
    sim = json.loads(Path(project_dir).joinpath("sim.json").read_text())
    cfg = sim.get("sims", {}).get("jv", {}).get("segment0", {}).get("config", {})
    return float(cfg.get("Vstart", 0.1)), float(cfg.get("Vstop", 4.0)), float(cfg.get("Vstep", 0.01))


def oghma_y_mesh_um(project: OghmaProject, ref_csv: Path | None = None) -> np.ndarray:
    """Return Oghma-compatible y positions in μm (from reference CSV if available)."""
    if ref_csv is None and project.outcoupling_dir is not None:
        ref_csv = project.outcoupling_dir / "photons_escape_prob_lam_avg_y.csv"
    if ref_csv is not None and Path(ref_csv).is_file():
        data = parse_oghma_csv(ref_csv)
        return _oghma_axis_to_um(data["x"], data["meta"], axis="y")

    return np.linspace(
        0.5 * project.y_mesh_len_um / project.y_mesh_points,
        project.total_thickness_um,
        project.y_mesh_points,
    )


def load_oghma_jv_reference(project_dir: Path | str) -> dict[str, Any]:
    """Load JV sweep CSV curves and sim_info summary from an Oghma project."""
    project_dir = Path(project_dir)
    out: dict[str, Any] = {}

    file_fields = {
        "jv.csv": {"jv_voltage": "x", "jv_current_density": "y"},
        "v_eqe.csv": {"v_eqe_voltage": "x", "v_eqe": "y"},
        "v_cd_m2.csv": {"v_luminance_voltage": "x", "v_luminance": "y"},
    }
    for fname, fields in file_fields.items():
        path = project_dir / fname
        if not path.is_file():
            continue
        d = parse_oghma_csv(path)
        for out_key, col in fields.items():
            out[out_key] = d[col]

    info_path = project_dir / "sim_info.dat"
    if info_path.is_file():
        out["sim_info"] = json.loads(info_path.read_text())
    return out


def list_oghma_snapshots(project_dir):
    def _parse(meta):
        return {"voltage": float(meta.get("voltage", np.nan))}

    return _list_indexed_snapshots(
        project_dir, "snapshots", parse_meta=_parse, require_meta=True
    )


def load_oghma_snapshot(project_dir, index):
    def _load(snap_dir, out):
        if "meta" in out:
            out["voltage"] = float(out["meta"].get("voltage", np.nan))
        for fname, keys in (
            ("eqe.csv", ("eqe_wl_um", "eqe")),
            ("luminescence_lam.csv", ("lum_wl_um", "luminescence")),
            ("emission_xyz.csv", ("gen_y_um", "photon_gen_rate")),
        ):
            path = snap_dir / fname
            if not path.is_file():
                continue
            d = parse_oghma_csv(path)
            meta = d["meta"]
            axis = "y" if fname == "eqe.csv" else "x"
            out[keys[0]] = _oghma_axis_to_um(d["x"], meta, axis=axis)
            out[keys[1]] = d["y"]

    return _load_indexed_snapshot(project_dir, "snapshots", index, load_files=_load)


def oghma_organic_y0_um(project: OghmaProject) -> float:
    """Full-stack y of first organic layer top (local y=0 in yl2 / emission_xyz)."""
    if len(project.layers) < 2:
        raise ValueError("project needs at least ITO + first organic layer")
    return float(project.layers[1].y0_um)


def load_oghma_emission_efficiency(project: OghmaProject) -> float:
    """Return pl_experimental_emission_efficiency_f2f from the emissive epitaxy layer."""
    eml = project.eml_layer()
    if eml is not None and eml.pl_efficiency_f2f is not None:
        return float(eml.pl_efficiency_f2f)
    raise ValueError(
        "EML pl_experimental_emission_efficiency_f2f missing in project; "
        "pass eta_pl explicitly or fix sim.json epitaxy."
    )


def load_oghma_photon_reference(project: OghmaProject) -> dict[str, Any]:
    """Load Oghma photon-density heatmap (photons_yl.csv)."""
    if project.outcoupling_dir is None:
        return {}
    path = project.outcoupling_dir / "photons_yl.csv"
    if not path.is_file():
        return {}
    heat = parse_oghma_csv(path)
    wl = _oghma_axis_to_um(heat["x"], heat["meta"], axis="y")
    y_um = _oghma_axis_to_um(heat["y"], heat["meta"], axis="y")
    density = heat["z"]
    wl_unique = np.unique(wl)
    y_unique = np.unique(y_um)
    grid = np.full((len(wl_unique), len(y_unique)), np.nan)
    wl_to_i = {w: i for i, w in enumerate(wl_unique)}
    y_to_i = {y: i for i, y in enumerate(y_unique)}
    for w, y, v in zip(wl, y_um, density):
        grid[wl_to_i[w], y_to_i[y]] = v
    return {"heat_wl_um": wl_unique, "heat_y_um": y_unique, "photon_density": grid}


def load_oghma_escape_reference(project: OghmaProject) -> dict[str, Any]:
    """Load Oghma outcoupling CSV reference curves."""
    if project.outcoupling_dir is None:
        return {}

    out: dict[str, Any] = {}
    heat_path = project.outcoupling_dir / "photons_escape_prob_yl.csv"
    lam_avg_y_path = project.outcoupling_dir / "photons_escape_prob_lam_avg_y.csv"
    y_avg_path = project.outcoupling_dir / "photons_escape_prob_y_avg.csv"

    if heat_path.is_file():
        heat = parse_oghma_csv(heat_path)
        wl = _oghma_axis_to_um(heat["x"], heat["meta"], axis="y")
        y_um = _oghma_axis_to_um(heat["y"], heat["meta"], axis="y")
        eta = heat["z"]
        wl_unique = np.unique(wl)
        y_unique = np.unique(y_um)
        grid = np.full((len(wl_unique), len(y_unique)), np.nan)
        wl_to_i = {w: i for i, w in enumerate(wl_unique)}
        y_to_i = {y: i for i, y in enumerate(y_unique)}
        for w, y, v in zip(wl, y_um, eta):
            grid[wl_to_i[w], y_to_i[y]] = v
        out["heat_wl_um"] = wl_unique
        out["heat_y_um"] = y_unique
        out["heat_eta"] = grid

    if lam_avg_y_path.is_file():
        d = parse_oghma_csv(lam_avg_y_path)
        out["lam_avg_y_um"] = _oghma_axis_to_um(d["x"], d["meta"], axis="y")
        out["lam_avg_eta"] = d["y"]

    if y_avg_path.is_file():
        d = parse_oghma_csv(y_avg_path)
        out["y_avg_wl_um"] = _oghma_axis_to_um(d["x"], d["meta"], axis="y")
        out["y_avg_eta"] = d["y"]

    if project.emission_spectrum_path and project.emission_spectrum_path.is_file():
        d = parse_oghma_csv(project.emission_spectrum_path)
        out["emission_wl_um"] = _oghma_axis_to_um(d["x"], d["meta"], axis="y")
        out["emission_prob"] = d["y"]

    out.update(load_oghma_photon_reference(project))
    return out


def compute_coupled_eqe_spectrum(
    *,
    wl_um: np.ndarray,
    escape_eta: np.ndarray,
    escape_y_um: np.ndarray,
    photon_gen_y_um: np.ndarray,
    photon_gen_rate: np.ndarray,
    emission_wl_um: np.ndarray,
    emission_prob: np.ndarray,
    ref_eqe_wl_um: np.ndarray | None = None,
    ref_eqe: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Coupled EQE spectrum: I(λ) ∝ S(λ) Σ_y G(y) η_esc(λ, y).

    escape_eta shape: (n_wl, n_y_esc)
    photon_gen_rate: 1D on photon_gen_y_um
    """
    wl_um = np.asarray(wl_um, dtype=float)
    escape_eta = np.asarray(escape_eta, dtype=float)
    escape_y_um = np.asarray(escape_y_um, dtype=float)
    photon_gen_y_um = np.asarray(photon_gen_y_um, dtype=float)
    photon_gen_rate = np.asarray(photon_gen_rate, dtype=float)
    emission_wl_um = np.asarray(emission_wl_um, dtype=float)
    emission_prob = np.asarray(emission_prob, dtype=float)

    g_on_esc_y = np.interp(escape_y_um, photon_gen_y_um, photon_gen_rate, left=0.0, right=0.0)
    g_sum = float(np.trapz(g_on_esc_y, escape_y_um))
    g_norm = g_on_esc_y / g_sum if g_sum > 0 else g_on_esc_y

    s_on_wl = np.interp(
        wl_um,
        emission_wl_um,
        emission_prob,
        left=float(emission_prob[0]) if emission_prob.size else 0.0,
        right=float(emission_prob[-1]) if emission_prob.size else 0.0,
    )
    # eqe_raw = s_on_wl * (escape_eta @ g_norm)
    eqe_raw = s_on_wl * np.trapz(escape_eta * g_norm[np.newaxis, :], escape_y_um, axis=1)

    eqe_aligned_to_reference = eqe_raw
    if ref_eqe is not None and ref_eqe_wl_um is not None:
        ref_on_wl = np.interp(wl_um, ref_eqe_wl_um, ref_eqe, left=0.0, right=0.0)
        ref_integral_scale = 1.0
        mask = ref_on_wl > 0
        if np.any(mask) and np.any(eqe_raw[mask] > 0):
            ref_integral_scale = float(np.sum(ref_on_wl[mask]) / np.sum(eqe_raw[mask]))
        eqe_aligned_to_reference = eqe_raw * ref_integral_scale

    result: dict[str, Any] = {
        "wl_um": wl_um,
        "eqe_raw": eqe_raw,
        "eqe_aligned_to_reference": eqe_aligned_to_reference,
        "eqe_norm": eqe_aligned_to_reference / max(float(eqe_aligned_to_reference.max()), 1e-30),
        "emission_on_wl": s_on_wl,
        "gen_profile_on_esc_y": g_norm,
    }

    if ref_eqe is not None and ref_eqe_wl_um is not None:
        ref_on_wl = np.interp(wl_um, ref_eqe_wl_um, ref_eqe, left=0.0, right=0.0)
        result["ref_eqe_on_wl"] = ref_on_wl
        result["ref_eqe_norm"] = ref_on_wl / max(float(ref_on_wl.max()), 1e-30)
        mask = np.isfinite(ref_on_wl) & np.isfinite(eqe_aligned_to_reference) & (ref_on_wl > 0)
        if np.any(mask):
            diff = eqe_aligned_to_reference[mask] - ref_on_wl[mask]
            result["rmse"] = float(np.sqrt(np.mean(diff**2)))
            result["corr"] = float(np.corrcoef(eqe_aligned_to_reference[mask], ref_on_wl[mask])[0, 1])
    return result


def build_eta_oghma_style_from_reference(
    wl_um: np.ndarray,
    y_um: np.ndarray,
    ref_escape: dict[str, Any],
) -> np.ndarray:
    """η_esc(λ,y) from Oghma formula: (heat_η / photon_density) × row_norm(photon_density)."""
    wl_arr = np.asarray(wl_um, dtype=float)
    y_arr = np.asarray(y_um, dtype=float)
    if "heat_eta" not in ref_escape:
        raise ValueError("ref_escape missing 'heat_eta'")
    ref_wl = np.asarray(ref_escape["heat_wl_um"], dtype=float)
    ref_y = np.asarray(ref_escape["heat_y_um"], dtype=float)
    ref_esc = np.asarray(ref_escape["heat_eta"], dtype=float)
    
    assert "photon_density" in ref_escape, "photon_density not in ref_escape"
    ref_ph = np.asarray(ref_escape["photon_density"], dtype=float)
    ratio_oghma = np.where(ref_ph > 1e-9, ref_esc / ref_ph, 0.0)
    ratio_on_grid = sample_on_ref_grid(ratio_oghma, ref_wl, ref_y, wl_arr, y_arr)
    ref_ph_on_grid = sample_on_ref_grid(ref_ph, ref_wl, ref_y, wl_arr, y_arr)
    ph_norm_for_eta = ref_ph_on_grid / np.maximum(
        ref_ph_on_grid.max(axis=1, keepdims=True), 1e-30
    )
    return ratio_on_grid * ph_norm_for_eta

def compute_oled_emission_outcoupling_maps_ito_al(
    project: OghmaProject,
    wl_um: np.ndarray,
    y_um: np.ndarray,
    simulation_module: Any,
    *,
    ref_escape: dict[str, Any] | None = None,
    escape_map: np.ndarray | None = None,
    photon_norm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build η_oghma_style outcoupling map (Oghma reference formula or future simulation inputs)."""
    _ = simulation_module
    wl_arr = np.asarray(wl_um, dtype=float)
    y_arr = np.asarray(y_um, dtype=float)
    if ref_escape is None:
        ref_escape = load_oghma_escape_reference(project)
    if escape_map is not None and photon_norm is not None:
        esc = np.asarray(escape_map, dtype=float)
        ph = np.asarray(photon_norm, dtype=float)
        ratio = np.where(ph > 1e-9, esc / ph, 0.0)
        ph_norm = ph / np.maximum(ph.max(axis=1, keepdims=True), 1e-30)
        eta_oghma_style = ratio * ph_norm
        source = "simulation"
    else:
        eta_oghma_style = build_eta_oghma_style_from_reference(wl_arr, y_arr, ref_escape)
        source = "oghma_reference"
    return {
        "eta_oghma_style": eta_oghma_style,
        "wl_um": wl_arr,
        "y_um": y_arr,
        "eta_oghma_style_source": source,
    }


# ---------------------------------------------------------------------------
# Emission alignment helpers (02_oled_emission_tmm)
# ---------------------------------------------------------------------------


def electrical_params_table(project: OghmaProject) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in project.layers:
        rows.append(
            {
                "layer": layer.name,
                "Xi (eV)": layer.xi_ev,
                "Eg (eV)": layer.eg_ev,
                "μe,y (m²/Vs)": layer.mue_y,
                "μh,y (m²/Vs)": layer.muh_y,
                "Nc/Nv (m⁻³)": layer.nc,
                "Gnp": layer.gnp,
                "k_f2f": layer.f2f_recomb,
            }
        )
    return rows


def _bilinear_on_regular_grid(
    values: np.ndarray,
    wl_um: np.ndarray,
    y_um: np.ndarray,
    wl_q: float,
    y_q: float,
) -> float:
    """Bilinear sample ``values[i_wl, i_y]`` on regular ``(wl_um, y_um)`` axes."""
    wl_um = np.asarray(wl_um, dtype=float)
    y_um = np.asarray(y_um, dtype=float)
    wl_q = float(np.clip(wl_q, wl_um[0], wl_um[-1]))
    y_q = float(np.clip(y_q, y_um[0], y_um[-1]))
    iw = int(np.clip(np.searchsorted(wl_um, wl_q, side="right") - 1, 0, len(wl_um) - 2))
    jy = int(np.clip(np.searchsorted(y_um, y_q, side="right") - 1, 0, len(y_um) - 2))
    wl0, wl1 = wl_um[iw], wl_um[iw + 1]
    y0, y1 = y_um[jy], y_um[jy + 1]
    tx = 0.0 if wl1 == wl0 else (wl_q - wl0) / (wl1 - wl0)
    ty = 0.0 if y1 == y0 else (y_q - y0) / (y1 - y0)
    v00 = values[iw, jy]
    v10 = values[iw + 1, jy]
    v01 = values[iw, jy + 1]
    v11 = values[iw + 1, jy + 1]
    return float(
        v00 * (1.0 - tx) * (1.0 - ty)
        + v10 * tx * (1.0 - ty)
        + v01 * (1.0 - tx) * ty
        + v11 * tx * ty
    )


def sample_on_ref_grid(
    values: np.ndarray,
    wl_um: np.ndarray,
    y_um: np.ndarray,
    ref_wl_um: np.ndarray,
    ref_y_um: np.ndarray,
) -> np.ndarray:
    """Bilinear sample a (n_wl, n_y) grid onto Oghma reference axes."""
    values = np.asarray(values, dtype=float)
    out = np.zeros((len(ref_wl_um), len(ref_y_um)), dtype=float)
    for i, wl in enumerate(ref_wl_um):
        for j, y in enumerate(ref_y_um):
            out[i, j] = _bilinear_on_regular_grid(values, wl_um, y_um, float(wl), float(y))
    return out


def peak_y_um(values_1d: np.ndarray, y_um: np.ndarray) -> float:
    """Return y position of maximum for a 1D depth profile."""
    return float(y_um[int(np.argmax(values_1d))])


def list_oghma_outcoupling_snapshots(project_dir):
    def _parse(meta):
        return {"wl_um": float(meta.get("wavelength", np.nan)) * 1e6}

    return _list_indexed_snapshots(project_dir, "outcoupling_snapshots", parse_meta=_parse)


def load_oghma_outcoupling_snapshot(project_dir, index):
    def _load(snap_dir, out):
        if "meta" in out:
            out["wl_um"] = float(out["meta"].get("wavelength", np.nan)) * 1e6
        for fname, key in (
            ("photons.csv", "photons"),
            ("photons_escape_prob.csv", "escape_prob"),
        ):
            path = snap_dir / fname
            if not path.is_file():
                continue
            d = parse_oghma_csv(path)
            out["y_um"] = _oghma_axis_to_um(d["x"], d["meta"], axis="x")
            out[key] = d["y"]

    return _load_indexed_snapshot(project_dir, "outcoupling_snapshots", index, load_files=_load)


def oled_ito_al_layer_depths_um(project: OghmaProject) -> list[float]:
    """Finite epitaxy depths (μm); semi-inf bookends are ``depth=0`` in the stack builder."""
    return [float(layer.thickness_um) for layer in project.layers]


def oled_emission_stack_labels_ito_al(project: OghmaProject) -> list[str]:
    """Bar labels for finite epitaxy layers (``ITO(∞)/Al(∞)`` are plot half-space arrows)."""
    return [
        f"{layer.name.split()[0]}(n={layer.optical_material.split('/')[-1]})"
        for layer in project.layers
    ]


def oled_emission_thicknesses_um_ito_al(project: OghmaProject) -> list[float]:
    """Bar-chart wrapper ``[0, *layer_depths*, 0]`` (legacy helper)."""
    depths = oled_ito_al_layer_depths_um(project)
    return [0.0, *depths, 0.0]


def build_oled_solver_from_project(
    project: OghmaProject,
    simulation_module: Any,
    *,
    ref_csv: Path | None = None,
) -> tuple[Any, np.ndarray]:
    """Build a configured C++ ``oled_device_solver`` from an Oghma project."""
    y_um = np.asarray(oghma_y_mesh_um(project, ref_csv=ref_csv), dtype=np.float64)
    layers_py = []
    for layer in project.layers:
        lp = simulation_module.oled_layer_params_d()
        lp.y0_um = float(layer.y0_um)
        lp.y1_um = float(layer.y1_um)
        lp.xi_ev = float(layer.xi_ev or 0.0)
        lp.eg_ev = float(layer.eg_ev or 0.0)
        lp.mue = float(layer.mue_y or 1e-5)
        lp.muh = float(layer.muh_y or 1e-5)
        lp.nc = float(layer.nc or 1e25)
        lp.nv = float(layer.nv or 1e25)
        lp.eps_r = 3.0
        lp.k_f2f = float(layer.f2f_recomb or 0.0)
        lp.eta_pl = float(layer.pl_efficiency_f2f or 0.0)
        lp.radiative = bool(layer.pl_emission_enabled)
        layers_py.append(lp)

    r_shunt = _read_oghma_rshunt_ohm_m2(project.project_dir)
    v_start, v_stop, v_step = _read_jv_sweep_config(project.project_dir)

    solver = simulation_module.oled_device_solver_d()
    solver.set_mesh(y_um)
    solver.set_layers(layers_py)
    solver.set_shunt_conductivity(1.0 / r_shunt)
    solver.set_sweep(v_start, v_stop, v_step)
    return solver, y_um


def _photon_gen_profile(recombination: np.ndarray, y_um: np.ndarray, eta_pl: float) -> np.ndarray:
    return np.asarray(recombination, dtype=float) * float(eta_pl)


def _optical_emission_current_density_a_m2(
    terminal_current_density_a_m2: float,
    voltage_v: float,
    shunt_conductivity_s_per_m2: float,
) -> float:
    """Radiative current for optics: max(J_terminal − g_shunt·V, 0). No sub-threshold floor."""
    j_after_shunt = float(terminal_current_density_a_m2) - float(shunt_conductivity_s_per_m2) * float(
        voltage_v
    )
    return max(j_after_shunt, 0.0)


def _eml_uniform_recombination_m3s(j_opt_a_m2: float, project: OghmaProject, y_um: np.ndarray) -> np.ndarray:
    """Uniform bimolecular source in the emissive layer matching ``j_opt/q``."""
    r = np.zeros_like(y_um, dtype=float)
    eml = project.eml_layer()
    if eml is None or j_opt_a_m2 <= 0.0:
        return r
    mask = (y_um >= eml.y0_um) & (y_um <= eml.y1_um)
    if not np.any(mask):
        return r
    thickness_m = max(float(eml.y1_um - eml.y0_um) * 1e-6, 1e-12)
    r[mask] = j_opt_a_m2 / ELEMENTARY_CHARGE_C / thickness_m
    return r


def _optical_recombination_profile_dd(
    J: float,
    v: float,
    project: OghmaProject,
    y_um: np.ndarray,
    g_shunt: float,
    *,
    r_native: np.ndarray | None = None,
) -> np.ndarray:
    """Optical R(y): prefer scaled native DD profile when self-consistent."""
    j_opt = _optical_emission_current_density_a_m2(J, v, g_shunt)
    j_target = j_opt / ELEMENTARY_CHARGE_C
    if r_native is not None:
        r_native = np.asarray(r_native, dtype=float)
        gen_int = float(np.trapz(r_native, y_um * 1e-6))
        if gen_int > 0.0 and j_target > 0.0:
            ratio = j_target / gen_int
            if MIN_NATIVE_RECOMB_PROFILE_SCALE_RATIO <= ratio <= MAX_NATIVE_RECOMB_PROFILE_SCALE_RATIO:
                return r_native * ratio
    return _eml_uniform_recombination_m3s(j_opt, project, y_um)


def _read_eml_srh_from_sim(project_dir: Path | str) -> dict[str, float | bool]:
    """Read SRH trap parameters from the EML epitaxy segment (dev use only)."""
    sim = json.loads(Path(project_dir).joinpath("sim.json").read_text())
    for seg in sim.get("epitaxy", {}).values():
        mat = str(seg.get("mat", "")).lower()
        if "alq3" in mat or str(seg.get("pl_emission_enabled", "")).lower() == "true":
            return {
                "srh_enabled": str(seg.get("ss_srh_enabled", "False")).lower() == "true",
                "srh_Nt_m3": float(seg.get("ss_srh_Nt", 0.0) or seg.get("Ntrape", 0.0) or 0.0),
                "srh_Etrap_ev": float(seg.get("ss_srh_trap_energy", 0.0) or seg.get("Etrape", 0.0) or 0.0),
                "srh_sigma_n_m2": float(seg.get("ss_srh_sigma_n", 1e-19)),
                "srh_sigma_p_m2": float(seg.get("ss_srh_sigma_p", 1e-19)),
            }
    return {
        "srh_enabled": False,
        "srh_Nt_m3": 0.0,
        "srh_Etrap_ev": 0.0,
        "srh_sigma_n_m2": 1e-19,
        "srh_sigma_p_m2": 1e-19,
    }


def _oled_enable_trap_from_env() -> bool:
    import os

    return os.environ.get("OLED_ENABLE_TRAP", "0").strip().lower() in {"1", "true", "yes"}


def _integrated_photon_gen(gen_rate: np.ndarray, y_um: np.ndarray) -> float:
    return float(np.trapz(np.asarray(gen_rate, dtype=float), np.asarray(y_um, dtype=float) * 1e-6))


def _eqe_percent_from_spectrum(wl_um: np.ndarray, eqe_density: np.ndarray) -> float:
    return float(np.trapz(np.asarray(eqe_density, dtype=float), np.asarray(wl_um, dtype=float)))


def _scalar_eqe_percent_from_coupled(
    coupled: dict[str, Any],
    *,
    gen_integral_m2_inv_s: float,
    current_density_a_m2: float,
) -> float:
    """
    Integrated external quantum efficiency (%).

    ``eqe_raw`` uses a depth-normalized generation profile (spectral outcoupling shape);
    absolute EQE = 100 · (∫R·η_pl dy) · (∫eqe_raw dλ) / (J/e).
    """
    if gen_integral_m2_inv_s <= 0.0 or current_density_a_m2 <= 0.0:
        return 0.0
    electron_flux_m2_s = current_density_a_m2 / ELEMENTARY_CHARGE_C
    spectral_outcoupling = _eqe_percent_from_spectrum(coupled["wl_um"], coupled["eqe_raw"])
    return 100.0 * gen_integral_m2_inv_s * spectral_outcoupling / electron_flux_m2_s


def _cie_photopic_luminosity_v(wl_um: np.ndarray) -> np.ndarray:
    """CIE 1931 V(λ) on ``wl_um`` (relative photopic sensitivity, peak ≈ 1 at 555 nm)."""
    wl_um = np.asarray(wl_um, dtype=float)
    return np.interp(wl_um * 1e3, _CIE_1931_V_WL_NM, _CIE_1931_V_VALUES, left=0.0, right=0.0)


def _luminance_cd_m2_from_coupled_spectrum(
    wl_um: np.ndarray,
    eqe_spectral_density: np.ndarray,
    eqe_percent: float,
    current_density_a_m2: float,
) -> float:
    """
    Photopic luminance (cd/m²) from coupled EQE spectrum and terminal current.

    Photon flux ∝ (EQE/100)·J/e is distributed across λ by ``eqe_spectral_density`` shape;
    L = K_m · ∫ Φ_e,λ(λ)·V(λ) dλ with K_m = 683 lm/W (CIE 555 nm peak).
    """
    wl_um = np.asarray(wl_um, dtype=float)
    eqe_spec = np.asarray(eqe_spectral_density, dtype=float)
    if eqe_percent <= 0.0 or current_density_a_m2 <= 0.0:
        return 0.0
    spec_integral = float(np.trapz(eqe_spec, wl_um))
    if spec_integral <= 0.0:
        return 0.0

    spectral_shape = eqe_spec / spec_integral
    photon_flux_total_m2_s = (eqe_percent / 100.0) * current_density_a_m2 / ELEMENTARY_CHARGE_C
    photon_flux_per_nm = spectral_shape * photon_flux_total_m2_s

    wl_m = wl_um * 1e-6
    photon_energy_j = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S / wl_m
    radiant_flux_per_nm_w_m2 = photon_flux_per_nm * photon_energy_j

    photopic_v = _cie_photopic_luminosity_v(wl_um)
    return PHOTOPIC_PEAK_LUMINOUS_EFFICACY_lm_per_W * float(
        np.trapz(radiant_flux_per_nm_w_m2 * photopic_v, wl_um)
    )


def _prepare_oled_jv_optical_coupling_context(
    project: OghmaProject,
    simulation_module: Any,
    *,
    eta_esc_maps: dict[str, np.ndarray] | None,
    eta_pl: float | None | None,
    ref_escape: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shared escape-map and emission-spectrum setup for JV optical coupling."""
    if eta_pl is None:
        eta_pl = load_oghma_emission_efficiency(project)
    if ref_escape is None:
        ref_escape = load_oghma_escape_reference(project)

    mesh_y = oghma_y_mesh_um(project)
    map_wl_um = np.asarray(ref_escape.get("heat_wl_um", project.wl_um), dtype=float)
    map_y_um = np.asarray(ref_escape.get("heat_y_um", mesh_y), dtype=float)

    if eta_esc_maps is None:
        eta_esc_maps = compute_oled_emission_outcoupling_maps_ito_al(
            project,
            map_wl_um,
            map_y_um,
            simulation_module,
            ref_escape=ref_escape,
        )

    wl_um = np.asarray(eta_esc_maps.get("wl_um", map_wl_um), dtype=float)
    escape_eta = np.asarray(eta_esc_maps["eta_oghma_style"], dtype=float)
    escape_y_um = np.asarray(eta_esc_maps.get("y_um", map_y_um), dtype=float)
    emission_wl_um = np.asarray(ref_escape.get("emission_wl_um", wl_um), dtype=float)
    emission_prob = np.asarray(ref_escape.get("emission_prob", np.ones_like(emission_wl_um)), dtype=float)
    if emission_prob.size and emission_prob.max() > 0:
        emission_prob = emission_prob / emission_prob.max()

    return {
        "eta_pl": float(eta_pl),
        "eta_esc_maps": eta_esc_maps,
        "mesh_y": mesh_y,
        "wl_um": wl_um,
        "escape_eta": escape_eta,
        "escape_y_um": escape_y_um,
        "emission_wl_um": emission_wl_um,
        "emission_prob": emission_prob,
    }


def _scan_oled_jv_with_optical_coupling(
    voltage: np.ndarray,
    current_density: np.ndarray,
    *,
    mesh_y: np.ndarray,
    wl_um: np.ndarray,
    escape_eta: np.ndarray,
    escape_y_um: np.ndarray,
    emission_wl_um: np.ndarray,
    emission_prob: np.ndarray,
    eta_pl: float,
    eta_esc_maps: dict[str, np.ndarray],
    recombination_at_index: Callable[[int, float], tuple[np.ndarray, np.ndarray, float, float]],
) -> dict[str, Any]:
    """Scan JV bias points and attach spectrum-integrated EQE and photopic luminance."""
    v_eqe = np.zeros_like(voltage)
    v_luminance = np.zeros_like(voltage)
    eqe_spectra: list[dict[str, Any]] = []
    gen_integrals = np.zeros_like(voltage)

    for i, v in enumerate(voltage):
        _r_prof, gen_rate, gen_int, j_terminal = recombination_at_index(i, float(v))
        gen_integrals[i] = gen_int
        coupled = compute_coupled_eqe_spectrum(
            wl_um=wl_um,
            escape_eta=escape_eta,
            escape_y_um=escape_y_um,
            photon_gen_y_um=mesh_y,
            photon_gen_rate=gen_rate,
            emission_wl_um=emission_wl_um,
            emission_prob=emission_prob,
        )
        eqe_pct = _scalar_eqe_percent_from_coupled(
            coupled,
            gen_integral_m2_inv_s=gen_int,
            current_density_a_m2=j_terminal,
        )
        v_eqe[i] = eqe_pct
        v_luminance[i] = _luminance_cd_m2_from_coupled_spectrum(
            wl_um, coupled["eqe_raw"], eqe_pct, j_terminal
        )
        eqe_spectra.append(coupled)

    return {
        "voltage": voltage,
        "current_density": current_density,
        "v_eqe": v_eqe,
        "v_luminance": v_luminance,
        "eqe_spectra": eqe_spectra,
        "y_um": mesh_y,
        "wl_um": wl_um,
        "eta_esc_maps": eta_esc_maps,
        "eta_pl": float(eta_pl),
        "gen_integrals": gen_integrals,
    }


def _compute_oled_jv_optical_coupled_with_solver(
    project: OghmaProject,
    simulation_module: Any,
    *,
    build_solver: Callable[..., tuple[Any, np.ndarray]],
    use_dd_recombination: bool,
    eta_esc_maps: dict[str, np.ndarray] | None = None,
    eta_pl: float | None = None,
    ref_escape: dict[str, Any] | None = None,
    poisson_backend: str | None = None,
) -> dict[str, Any]:
    ctx = _prepare_oled_jv_optical_coupling_context(
        project,
        simulation_module,
        eta_esc_maps=eta_esc_maps,
        eta_pl=eta_pl,
        ref_escape=ref_escape,
    )
    if poisson_backend is None:
        solver, mesh_y = build_solver(project, simulation_module)
    else:
        solver, mesh_y = build_solver(
            project, simulation_module, poisson_backend=poisson_backend
        )
    solver.sweep_jv()
    voltage = np.array(solver.sweep_voltage(), dtype=float, copy=True)
    current_density = np.array(solver.sweep_current_density(), dtype=float, copy=True)

    if use_dd_recombination:
        g_shunt = 1.0 / _read_oghma_rshunt_ohm_m2(project.project_dir)
        sweep_r = (
            np.array(solver.sweep_recombination(), dtype=float, copy=True)
            if hasattr(solver, "sweep_recombination")
            else None
        )

        def recombination_at_index(i: int, v: float) -> tuple[np.ndarray, np.ndarray, float, float]:
            j_terminal = float(current_density[i])
            if sweep_r is not None and i < sweep_r.shape[0]:
                r_native = sweep_r[i]
            else:
                solver.solve_at_voltage(v)
                j_terminal = float(solver.current_density())
                r_native = np.array(solver.recombination_profile(), dtype=float, copy=True)
            r_prof = _optical_recombination_profile_dd(
                j_terminal, v, project, mesh_y, g_shunt, r_native=r_native
            )
            gen_rate = _photon_gen_profile(r_prof, mesh_y, ctx["eta_pl"])
            gen_int = _integrated_photon_gen(gen_rate, mesh_y)
            return r_prof, gen_rate, gen_int, j_terminal
    else:

        def recombination_at_index(i: int, v: float) -> tuple[np.ndarray, np.ndarray, float, float]:
            solver.solve_at_voltage(v)
            r_prof = np.array(solver.recombination_profile(), dtype=float, copy=True)
            gen_rate = _photon_gen_profile(r_prof, mesh_y, ctx["eta_pl"])
            gen_int = _integrated_photon_gen(gen_rate, mesh_y)
            return r_prof, gen_rate, gen_int, float(current_density[i])

    return _scan_oled_jv_with_optical_coupling(
        voltage,
        current_density,
        mesh_y=mesh_y,
        wl_um=ctx["wl_um"],
        escape_eta=ctx["escape_eta"],
        escape_y_um=ctx["escape_y_um"],
        emission_wl_um=ctx["emission_wl_um"],
        emission_prob=ctx["emission_prob"],
        eta_pl=ctx["eta_pl"],
        eta_esc_maps=ctx["eta_esc_maps"],
        recombination_at_index=recombination_at_index,
    )


def compute_oled_jv_optical_coupled(
    project: OghmaProject,
    simulation_module: Any,
    *,
    eta_esc_maps: dict[str, np.ndarray] | None = None,
    eta_pl: float | None = None,
    ref_escape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Full JV / V-EQE / V-L scan with emission-TMM outcoupling.

    Returns voltage/current arrays plus ``v_eqe``, ``v_luminance``, and per-voltage spectra.
    """
    return _compute_oled_jv_optical_coupled_with_solver(
        project,
        simulation_module,
        build_solver=build_oled_solver_from_project,
        use_dd_recombination=False,
        eta_esc_maps=eta_esc_maps,
        eta_pl=eta_pl,
        ref_escape=ref_escape,
    )


def _poisson_backend_from_str(name: str | None) -> int:
    import os

    key = (name or os.environ.get("OLED_POISSON_BACKEND", "thomas")).strip().lower()
    return 1 if key in {"sl", "sturm_liouville", "sturm-liouville"} else 0


def _read_oghma_dd_math(project_dir: Path | str) -> dict[str, float | int]:
    sim = json.loads(Path(project_dir).joinpath("sim.json").read_text())
    math = sim.get("math", {})
    return {
        "electricalerror": float(math.get("electricalerror", 1e-8)),
        "electricalclamp": float(math.get("electricalclamp", 1.0)),
        "maxelectricalitt": int(math.get("maxelectricalitt", 100)),
        "minelectricalitt": int(math.get("minelectricalitt", 1)),
    }


def _read_oghma_contact(
    project_dir: Path | str,
    layer_segment_key: str,
    *,
    contacts_segment_key: str | None = None,
    sim: dict | None = None,
) -> dict[str, float | bool]:
    if sim is None:
        sim = json.loads(Path(project_dir).joinpath("sim.json").read_text())
    seg = sim.get("epitaxy", {}).get(layer_segment_key, {})
    shape = seg.get("shape_electrical", {})
    ckey = contacts_segment_key if contacts_segment_key is not None else layer_segment_key
    contact = sim.get("epitaxy", {}).get("contacts", {}).get(ckey, {}).get("contact", {})
    majority = str(contact.get("majority_model", "ohmic")).lower()
    minority = str(contact.get("minority_model", "blocking")).lower()
    majority_carrier = str(contact.get("majority", "")).lower()
    if not majority_carrier:
        majority_carrier = "hole" if layer_segment_key == "segment0" else "electron"
    component = str(shape.get("electrical_component", "resistance")).lower()
    return {
        "j0": float(shape.get("electrical_J0", 5e-9)),
        "n": float(shape.get("electrical_n", 1.2)),
        "series_y": float(shape.get("electrical_series_y", 0.0)),
        "use_resistance": component == "resistance",
        "majority_ohmic": majority == "ohmic",
        "minority_blocking": minority == "blocking",
        "majority_is_hole": majority_carrier == "hole",
    }


def _build_oled_dd_layers_and_mesh(
    project: OghmaProject,
    simulation_module: Any,
    *,
    ref_csv: Path | None = None,
) -> tuple[list[Any], np.ndarray]:
    y_um = np.asarray(oghma_y_mesh_um(project, ref_csv=ref_csv), dtype=np.float64)
    trap_params = _read_eml_srh_from_sim(project.project_dir) if _oled_enable_trap_from_env() else None
    layers_py = []
    for layer in project.layers:
        lp = simulation_module.oled_layer_params_d()
        lp.y0_um = float(layer.y0_um)
        lp.y1_um = float(layer.y1_um)
        lp.xi_ev = float(layer.xi_ev or 0.0)
        lp.eg_ev = float(layer.eg_ev or 0.0)
        lp.mue = float(layer.mue_y or 1e-5)
        lp.muh = float(layer.muh_y or 1e-5)
        lp.nc = float(layer.nc or 1e25)
        lp.nv = float(layer.nv or 1e25)
        lp.eps_r = 3.0
        lp.k_f2f = float(layer.f2f_recomb or 0.0)
        lp.eta_pl = float(layer.pl_efficiency_f2f or 0.0)
        lp.radiative = bool(layer.pl_emission_enabled)
        if trap_params is not None and lp.k_f2f > 0.0:
            lp.srh_enabled = bool(trap_params["srh_enabled"]) or trap_params["srh_Nt_m3"] > 0.0
            lp.srh_Nt_m3 = float(trap_params["srh_Nt_m3"])
            lp.srh_Etrap_ev = float(trap_params["srh_Etrap_ev"])
            lp.srh_sigma_n_m2 = float(trap_params["srh_sigma_n_m2"])
            lp.srh_sigma_p_m2 = float(trap_params["srh_sigma_p_m2"])
        layers_py.append(lp)
    return layers_py, y_um


def _configure_oled_dd_solver_common(
    solver: Any,
    project: OghmaProject,
    simulation_module: Any,
    *,
    ref_csv: Path | None = None,
    poisson_backend: str | None = None,
) -> tuple[Any, np.ndarray]:
    layers_py, y_um = _build_oled_dd_layers_and_mesh(project, simulation_module, ref_csv=ref_csv)
    r_shunt = _read_oghma_rshunt_ohm_m2(project.project_dir)
    v_start, v_stop, v_step = _read_jv_sweep_config(project.project_dir)
    sim = json.loads(Path(project.project_dir).joinpath("sim.json").read_text())
    epitaxy = sim.get("epitaxy", {})
    seg_keys = [k for k in epitaxy if k.startswith("segment") and k.replace("segment", "").isdigit()]
    seg_keys.sort(key=lambda k: int(k.replace("segment", "")))
    cathode_layer_key = seg_keys[-1] if seg_keys else "segment0"
    anode = _read_oghma_contact(
        project.project_dir, "segment0", contacts_segment_key="segment0", sim=sim
    )
    cathode = _read_oghma_contact(
        project.project_dir,
        cathode_layer_key,
        contacts_segment_key="segment1",
        sim=sim,
    )

    contact_a = simulation_module.oled_dd_contact_params_d()
    contact_a.j0_a_m2 = anode["j0"]
    contact_a.ideality_n = anode["n"]
    contact_a.series_r_y_ohm_m2 = anode["series_y"]
    contact_a.use_resistance = bool(anode["use_resistance"])
    contact_a.majority_ohmic = bool(anode["majority_ohmic"])
    contact_a.minority_blocking = bool(anode["minority_blocking"])
    contact_a.majority_is_hole = bool(anode["majority_is_hole"])
    contact_c = simulation_module.oled_dd_contact_params_d()
    contact_c.j0_a_m2 = cathode["j0"]
    contact_c.ideality_n = cathode["n"]
    contact_c.series_r_y_ohm_m2 = cathode["series_y"]
    contact_c.use_resistance = bool(cathode["use_resistance"])
    contact_c.majority_ohmic = bool(cathode["majority_ohmic"])
    contact_c.minority_blocking = bool(cathode["minority_blocking"])
    contact_c.majority_is_hole = bool(cathode["majority_is_hole"])

    solver.set_mesh(y_um)
    solver.set_layers(layers_py)
    solver.set_shunt_conductivity(1.0 / r_shunt)
    solver.set_sweep(v_start, v_stop, v_step)
    solver.set_contacts(contact_a, contact_c)
    if hasattr(solver, "set_poisson_backend"):
        solver.set_poisson_backend(_poisson_backend_from_str(poisson_backend))
    if hasattr(solver, "set_enable_srh_recombination") and _oled_enable_trap_from_env():
        solver.set_enable_srh_recombination(True)
    return solver, y_um


def build_oled_gummel_solver_from_project(
    project: OghmaProject,
    simulation_module: Any,
    *,
    ref_csv: Path | None = None,
    poisson_backend: str | None = None,
) -> tuple[Any, np.ndarray]:
    solver = simulation_module.oled_gummel_dd_solver_d()
    gummel = simulation_module.oled_dd_gummel_settings_d()
    gummel.max_iter = 120
    gummel.tol = 50.0
    gummel.damping = 0.4
    solver.set_gummel_settings(gummel)
    return _configure_oled_dd_solver_common(
        solver, project, simulation_module, ref_csv=ref_csv, poisson_backend=poisson_backend
    )


def build_oled_newton_solver_from_project(
    project: OghmaProject,
    simulation_module: Any,
    *,
    ref_csv: Path | None = None,
    poisson_backend: str | None = None,
) -> tuple[Any, np.ndarray]:
    solver = simulation_module.oled_newton_dd_solver_d()
    math = _read_oghma_dd_math(project.project_dir)
    newton = simulation_module.oled_dd_newton_settings_d()
    newton.max_iter = int(math["maxelectricalitt"])
    newton.min_iter = int(math["minelectricalitt"])
    newton.tol = float(math["electricalerror"])
    newton.clamp = float(math["electricalclamp"])
    solver.set_newton_settings(newton)
    return _configure_oled_dd_solver_common(
        solver, project, simulation_module, ref_csv=ref_csv, poisson_backend=poisson_backend
    )


def compute_oled_jv_optical_coupled_gummel(
    project: OghmaProject,
    simulation_module: Any,
    *,
    eta_esc_maps: dict[str, np.ndarray] | None = None,
    eta_pl: float | None = None,
    ref_escape: dict[str, Any] | None = None,
    poisson_backend: str | None = None,
) -> dict[str, Any]:
    return _compute_oled_jv_optical_coupled_with_solver(
        project,
        simulation_module,
        build_solver=build_oled_gummel_solver_from_project,
        use_dd_recombination=True,
        eta_esc_maps=eta_esc_maps,
        eta_pl=eta_pl,
        ref_escape=ref_escape,
        poisson_backend=poisson_backend,
    )


def compute_oled_jv_optical_coupled_newton(
    project: OghmaProject,
    simulation_module: Any,
    *,
    eta_esc_maps: dict[str, np.ndarray] | None = None,
    eta_pl: float | None = None,
    ref_escape: dict[str, Any] | None = None,
    poisson_backend: str | None = None,
) -> dict[str, Any]:
    return _compute_oled_jv_optical_coupled_with_solver(
        project,
        simulation_module,
        build_solver=build_oled_newton_solver_from_project,
        use_dd_recombination=True,
        eta_esc_maps=eta_esc_maps,
        eta_pl=eta_pl,
        ref_escape=ref_escape,
        poisson_backend=poisson_backend,
    )



# ---------------------------------------------------------------------------
# Re-exports for notebooks (import oghma_core / oghma_fabry directly in new code)
# ---------------------------------------------------------------------------

from oghma_core import (  # noqa: E402
    build_oghma_passive_stack,
    compare_metrics,
    fdtd_input_combine_power,
    filter_stack_layers,
    list_oghma_optical_snapshots,
    load_oghma_optical_reference,
    load_oghma_optical_snapshot,
    load_oghma_project,
    normalize_rows,
    passive_filter_stack_labels,
)
from oghma_fabry import (  # noqa: E402
    build_fabry_perot_stack,
    compute_fabry_emission_case_bc_spectra,
    compute_fabry_emission_transfer,
    compute_fabry_lam_e_norm,
    compute_fabry_passive_transfer,
    compute_fabry_reflection_ahead_of_z,
    fabry_perot_interface_R_at_wl,
    fabry_perot_stack_labels,
    fabry_perot_thicknesses_um,
    parse_fabry_perot_geometry,
    plot_fabry_perot_fdtd_stack_section,
)
from oghma_bragg import (  # noqa: E402
    build_bragg_fdtd_stack,
    compute_bragg_emission_transfer,
    compute_bragg_lam_e_norm,
    compute_bragg_passive_transfer,
    compute_bragg_reflection_ahead_of_z,
    parse_bragg_grating_geometry,
)
from oghma_fdtd_alignment import (  # noqa: E402
    display_alignment_reports,
    evaluate_case,
    load_fdtd_alignment_bundle,
    plot_case_summary_grid,
    plot_forward_source_diagnostic,
    s_in_times_intensity,
)
from oghma_fdtd_source import (  # noqa: E402
    integrate_incoherent_power,
    predict_fdtd_output_spectrum_incoherent,
)


def air_layer_thickness_nm(project: OghmaProject) -> float:
    from oghma_core import air_layer_thickness_um

    return air_layer_thickness_um(project) * 1000.0


def fabry_perot_thicknesses_nm(geom):  # noqa: ANN001
    return [t * 1000.0 for t in fabry_perot_thicknesses_um(geom)]


def oghma_y_mesh_nm(project: OghmaProject, ref_csv: Path | None = None) -> np.ndarray:
    return oghma_y_mesh_um(project, ref_csv=ref_csv) * 1000.0
