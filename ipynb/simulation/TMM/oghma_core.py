"""Core Oghma materials, project loading, optical reference, metrics, and TMM helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

import simulation  # noqa: F401 — must load before simulation_database_parser

SUBSTRATE_DEPTH_UM = 1000.0


def _load_coating_funcs():
    import simulation  # noqa: F401 — PYTHONPATH must include SIMULATION_ARTIFACTS_DIR
    from coating_visualizer import build_tmm_layers, layers_from_formula

    return build_tmm_layers, layers_from_formula


def _load_build_tmm_layers():
    return _load_coating_funcs()[0]


def _tmm_layers_from_project_formula(
    formula: str,
    materials_db: dict[str, Any],
    lam_um: float,
    simulation_module: Any,
) -> list[Any]:
    del lam_um
    build_tmm_layers, layers_from_formula = _load_coating_funcs()
    materials, thicknesses_um = layers_from_formula(
        formula, materials_db, simulation_module=simulation_module
    )
    return build_tmm_layers(materials, thicknesses_um, simulation_module=simulation_module)


def _tmm_layers_from_thicknesses_um(
    materials: Sequence[Any],
    thicknesses_um: Sequence[float],
    simulation_module: Any,
) -> list[Any]:
    build_tmm_layers = _load_build_tmm_layers()
    return build_tmm_layers(
        list(materials),
        list(thicknesses_um),
        simulation_module=simulation_module,
    )


def _tmm_layers_from_nk_thicknesses_um(
    nk_list: Sequence[complex],
    thicknesses_um: Sequence[float],
    simulation_module: Any,
) -> list[Any]:
    from coating_solver_test_util import make_coating

    return [
        make_coating(simulation_module, complex(nk), float(d), "")
        for nk, d in zip(nk_list, thicknesses_um)
    ]


@lru_cache(maxsize=256)
def _read_oghma_material(material_path: str) -> Any:
    """Load material via simulation_database (sole supported resolution path)."""
    from oghma_runtime import artifacts_cwd

    sdp = _simulation_database_parser()
    db = sdp.get_simulation_database(init=True)
    with artifacts_cwd():
        return sdp.read_at_query_path(db, sdp.material_query_keys(material_path))


@lru_cache(maxsize=64)
def _read_oghma_spectrum(spectrum_path: str) -> Any:
    """Load spectrum via simulation_database."""
    from oghma_runtime import artifacts_cwd

    sdp = _simulation_database_parser()
    db = sdp.get_simulation_database(init=True)
    with artifacts_cwd():
        return sdp.read_at_query_path(db, sdp.spectrum_query_keys(spectrum_path))


def preload_oghma_materials(
    material_paths: Iterable[str],
) -> list[str]:
    """Eagerly load materials via simulation_database."""
    loaded: list[str] = []
    for material_path in dict.fromkeys(material_paths):
        _read_oghma_material(material_path)
        loaded.append(material_path)
    return loaded


def load_oghma_material_nk(
    material_path: str,
    wl_um: float | np.ndarray,
) -> complex | np.ndarray:
    """Load dispersive nk via simulation_database (cached read + nk_at)."""
    from coating_solver_test_util import material_nk_at_wl

    mat = _read_oghma_material(material_path)
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    nk = np.asarray(
        [material_nk_at_wl(mat, float(w)) for w in wl_arr],
        dtype=complex,
    )
    if np.ndim(wl_um) == 0:
        return complex(nk[0])
    return nk


def find_emission_spectrum_path(project_dir: Path | str) -> Path | None:
    """Discover emission_input*.csv in an Oghma project directory."""
    project_dir = Path(project_dir)
    matches = sorted(project_dir.glob("emission_input*.csv"))
    return matches[0] if matches else None


@dataclass
class OghmaLayer:
    name: str
    y0_um: float
    thickness_um: float
    optical_material: str
    obj_type: str = "other"
    pl_emission_enabled: bool = False
    pl_input_spectrum: str | None = None
    pl_efficiency_f2f: float | None = None
    xi_ev: float | None = None
    eg_ev: float | None = None
    mue_y: float | None = None
    muh_y: float | None = None
    nc: float | None = None
    nv: float | None = None
    gnp: float | None = None
    f2f_recomb: float | None = None

    @property
    def y1_um(self) -> float:
        return self.y0_um + self.thickness_um

    @property
    def is_emissive(self) -> bool:
        return self.pl_emission_enabled


@dataclass
class OghmaProject:
    project_dir: Path
    simmode: str
    version: str
    layers: list[OghmaLayer]
    wl_start_um: float
    wl_stop_um: float
    wl_points: int
    y_mesh_len_um: float
    y_mesh_points: int
    emission_spectrum_path: Path | None = None
    outcoupling_dir: Path | None = None
    light_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_thickness_um(self) -> float:
        if not self.layers:
            return 0.0
        return max(layer.y1_um for layer in self.layers)

    @property
    def wl_um(self) -> np.ndarray:
        return np.linspace(self.wl_start_um, self.wl_stop_um, int(self.wl_points))

    def eml_layer(self) -> OghmaLayer | None:
        for layer in self.layers:
            if layer.pl_emission_enabled:
                return layer
        for layer in self.layers:
            if "eml" in layer.name.lower() or "alq3" in layer.name.lower():
                return layer
        return None

    def stack_top_to_bottom(self) -> list[OghmaLayer]:
        """Notebook z-axis order: cathode (top) -> anode/substrate (bottom)."""
        return list(reversed(self.layers))


def parse_oghma_csv(path: Path | str) -> dict[str, Any]:
    """Parse Oghma #oghma_csv file into metadata and numeric arrays."""
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty Oghma CSV: {path}")

    header = lines[0]
    meta: dict[str, Any] = {}
    if header.startswith("#oghma_csv"):
        match = re.search(r"\{(.+)\}", header)
        if match:
            meta = json.loads("{" + match.group(1) + "}")

    rows: list[list[float]] = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(x) for x in line.replace(",", " ").split()]
        rows.append(parts)

    arr = np.asarray(rows, dtype=float)
    result: dict[str, Any] = {"meta": meta, "raw": arr, "path": path}

    cols = str(meta.get("cols", "yd"))
    file_type = str(meta.get("type", "xy"))
    if arr.size == 0:
        return result

    if file_type == "heat" and arr.shape[1] >= 3:
        result["x"] = arr[:, 0]
        result["y"] = arr[:, 1]
        result["z"] = arr[:, 2]
    elif cols == "yd" and arr.shape[1] >= 2:
        result["x"] = arr[:, 0]
        result["y"] = arr[:, 1]
    elif cols == "xyd" and arr.shape[1] >= 3:
        result["x"] = arr[:, 0]
        result["y"] = arr[:, 1]
        result["z"] = arr[:, 2]
    else:
        result["x"] = arr[:, 0]
        if arr.shape[1] > 1:
            result["y"] = arr[:, 1]
        if arr.shape[1] > 2:
            result["z"] = arr[:, 2]
    return result


def _oghma_axis_to_um(
    values: np.ndarray,
    meta: dict[str, Any],
    *,
    axis: str = "x",
    heatmap: bool = False,
) -> np.ndarray:
    """Convert Oghma CSV axis column to μm."""
    if heatmap:
        return np.asarray(values, dtype=float) * 1e6
    mul_key = "y_mul" if axis == "y" else "x_mul"
    units_key = "y_units" if axis == "y" else "x_units"
    fallback_mul = meta.get("y_mul" if axis == "x" else "x_mul", 1.0)
    fallback_units = meta.get("y_units" if axis == "x" else "x_units", "m")
    mul = float(meta.get(mul_key, fallback_mul))
    units = str(meta.get(units_key, fallback_units))
    pos = values * mul if mul != 1.0 else values
    if units == "nm":
        return np.asarray(pos, dtype=float) * 1e-3
    if units == "um":
        return np.asarray(pos, dtype=float)
    return np.asarray(pos, dtype=float) * 1e6


def _simulation_database_parser():
    """Requires ``import simulation`` first (PYTHONPATH must include artifacts)."""
    import simulation_database_parser as sdp

    return sdp


def load_optical_config(project_dir: Path | str) -> dict[str, Any]:
    """Return Oghma optical boundary and mesh settings from sim.json."""
    project_dir = Path(project_dir)
    sim = json.loads((project_dir / "sim.json").read_text())
    opt = sim.get("optical", {})
    boundary = opt.get("boundary", {})
    mesh = opt.get("mesh", {})

    def _mesh_segment(key: str) -> dict[str, Any]:
        seg = mesh.get(key, {}).get("segment0", {})
        out = dict(seg)
        for field in ("start", "stop", "len"):
            val = out.get(field)
            if val is None:
                continue
            fval = float(val)
            units = str(out.get(f"{field}_u", "m"))
            if fval < 1e-5:
                out[f"{field}_nm"] = fval * 1e9
            elif units == "nm":
                out[f"{field}_nm"] = fval
            else:
                out[f"{field}_nm"] = fval * 1e9
        return out

    return {
        "boundary_y0": str(boundary.get("optical_y0", "")),
        "boundary_y1": str(boundary.get("optical_y1", "")),
        "mesh_l": _mesh_segment("mesh_l"),
        "mesh_y": _mesh_segment("mesh_y"),
    }


def _ray_trace_theta_from_shape_pl(sim: dict[str, Any]) -> dict[str, Any]:
    """PL-layer ``ray_theta_*`` fields (ray-tracing only, not emission-TMM u sampling)."""
    epi = sim.get("epitaxy", {})
    n_seg = int(epi.get("segments", 0))
    for i in range(n_seg):
        seg = epi.get(f"segment{i}", {})
        pl = seg.get("shape_pl", {})
        if str(pl.get("pl_emission_enabled", "False")).lower() == "true":
            return {
                "ray_theta_deg": float(pl.get("ray_theta", 0.0)),
                "ray_theta_min_deg": float(
                    pl.get("ray_theta_min", pl.get("ray_theta_start", 0.0))
                ),
                "ray_theta_max_deg": float(
                    pl.get("ray_theta_max", pl.get("ray_theta_stop", 0.0))
                ),
                "ray_theta_steps": int(pl.get("ray_theta_steps", 1)),
            }
    return {
        "ray_theta_deg": 0.0,
        "ray_theta_min_deg": 0.0,
        "ray_theta_max_deg": 0.0,
        "ray_theta_steps": 1,
    }


def load_outcoupling_config(project_dir: Path | str) -> dict[str, Any]:
    """Oghma optical mesh, outcoupling model, and emission angle settings from sim.json."""
    project_dir = Path(project_dir)
    sim = json.loads((project_dir / "sim.json").read_text())
    opt = sim.get("optical", {})
    outc = opt.get("outcoupling", {})
    light_seg = opt.get("light_sources", {}).get("lights", {}).get("segment0", {})
    light_src = light_seg.get("light_source", light_seg)
    illuminate_from = str(
        light_src.get("light_illuminate_from", light_seg.get("light_illuminate_from", ""))
    )
    cfg = load_optical_config(project_dir)
    cfg.update(
        {
            "outcoupling_model": str(outc.get("outcoupling_model", "")),
            "incoherent_wavelengths": int(outc.get("incoherent_wavelengths", 1)),
            "light_illuminate_from": illuminate_from,
            **_ray_trace_theta_from_shape_pl(sim),
        }
    )
    return cfg


def oghma_emission_u_from_ray_theta_deg(theta_deg: float) -> float:
    """TMM normalized in-plane wavenumber ``u = n sin(theta)`` (air incidence → ``n≈1``)."""
    return math.sin(math.radians(float(theta_deg)))


def oghma_emission_u_values_from_project(project: OghmaProject) -> list[float]:
    """In-plane ``u`` for emission-TMM outcoupling (``transfer_matrix`` → normal ``u=0``).

    ``shape_pl.ray_theta_*`` is a ray-tracing grid in Oghma/gpvdm; it must not drive
    transfer-matrix outcoupling angle sampling (see ``00_software_alignment_skills.md``).
    """
    cfg = load_outcoupling_config(project.project_dir)
    model = str(cfg.get("outcoupling_model", "")).strip().lower()
    if model in ("transfer_matrix", "unity", "off", ""):
        return [0.0]
    if model != "ray_trace":
        return [0.0]
    theta_deg = float(cfg.get("ray_theta_deg", 0.0))
    steps = int(cfg.get("ray_theta_steps", 1))
    t_min = float(cfg.get("ray_theta_min_deg", theta_deg))
    t_max = float(cfg.get("ray_theta_max_deg", theta_deg))
    if steps <= 1 or t_min == t_max:
        return [oghma_emission_u_from_ray_theta_deg(theta_deg)]
    thetas = np.linspace(t_min, t_max, steps)
    return [oghma_emission_u_from_ray_theta_deg(t) for t in thetas]


def epitaxy_table(project: OghmaProject) -> list[dict[str, Any]]:
    """Return epitaxy layers bottom→top (Oghma y=0 at ITO anode)."""
    rows: list[dict[str, Any]] = []
    for layer in project.layers:
        rows.append(
            {
                "layer": layer.name,
                "obj_type": layer.obj_type,
                "y0_um": layer.y0_um,
                "y1_um": layer.y1_um,
                "t_um": layer.thickness_um,
                "optical_material": layer.optical_material,
                "pl_emission": layer.pl_emission_enabled,
                "pl_spectrum": layer.pl_input_spectrum or "",
            }
        )
    return rows


def load_oghma_project(project_dir: Path | str) -> OghmaProject:
    project_dir = Path(project_dir)
    sim = json.loads((project_dir / "sim.json").read_text())

    epi = sim["epitaxy"]
    n_segments = int(epi.get("segments", 0))
    layers: list[OghmaLayer] = []
    for i in range(n_segments):
        seg = epi[f"segment{i}"]
        dos = seg.get("shape_dos", {})
        pl = seg.get("shape_pl", {})
        pl_enabled = str(pl.get("pl_emission_enabled", "False")).lower() == "true"
        pl_spectrum = pl.get("pl_input_spectrum")
        pl_spectrum = None if pl_spectrum in (None, "", "none") else str(pl_spectrum)
        pl_f2f = pl.get("pl_experimental_emission_efficiency_f2f")
        layers.append(
            OghmaLayer(
                name=str(seg.get("name", f"layer{i}")),
                y0_um=float(seg["y0"]) * 1e6,
                thickness_um=float(seg["dy"]) * 1e6,
                optical_material=str(seg.get("optical_material", "")),
                obj_type=str(seg.get("obj_type", "other")),
                pl_emission_enabled=pl_enabled,
                pl_input_spectrum=pl_spectrum,
                pl_efficiency_f2f=float(pl_f2f) if pl_f2f is not None else None,
                xi_ev=float(dos["Xi"]) if dos.get("Xi") is not None else None,
                eg_ev=float(dos["Eg"]) if dos.get("Eg") is not None else None,
                mue_y=float(dos["mue_y"]) if dos.get("mue_y") is not None else None,
                muh_y=float(dos["muh_y"]) if dos.get("muh_y") is not None else None,
                nc=float(dos["Nc"]) if dos.get("Nc") is not None else None,
                nv=float(dos["Nv"]) if dos.get("Nv") is not None else None,
                gnp=float(seg.get("Gnp", 0.0)),
                f2f_recomb=float(dos.get("free_to_free_recombination", 0.0)),
            )
        )

    opt_mesh = sim.get("optical", {}).get("mesh", {})
    mesh_l = opt_mesh.get("mesh_l", {}).get("segment0", {})
    mesh_y = opt_mesh.get("mesh_y", {}).get("segment0", {})

    wl_start = float(mesh_l.get("start", 300e-9))
    wl_stop = float(mesh_l.get("stop", 580e-9))
    # Oghma mesh_l often labels "nm" while values remain in metres (e.g. 3e-07 = 300 nm).
    if wl_start < 1e-5:
        wl_start_um = wl_start * 1e6
        wl_stop_um = wl_stop * 1e6
    elif str(mesh_l.get("start_u", "m")) == "nm":
        wl_start_um = wl_start * 1e-3
        wl_stop_um = wl_stop * 1e-3
    else:
        wl_start_um = wl_start * 1e6
        wl_stop_um = wl_stop * 1e6

    emission_path = find_emission_spectrum_path(project_dir)
    outcoupling_dir = project_dir / "outcoupling"
    light_stats_path = project_dir / "light_stats.json"
    light_stats = json.loads(light_stats_path.read_text()) if light_stats_path.is_file() else {}

    return OghmaProject(
        project_dir=project_dir,
        simmode=str(sim.get("sim", {}).get("simmode", "")),
        version=str(sim.get("sim", {}).get("version", "")),
        layers=layers,
        wl_start_um=wl_start_um,
        wl_stop_um=wl_stop_um,
        wl_points=int(mesh_l.get("points", 10)),
        y_mesh_len_um=float(mesh_y.get("len", 5.2e-7)) * 1e6,
        y_mesh_points=int(mesh_y.get("points", 200)),
        emission_spectrum_path=emission_path if emission_path is not None and emission_path.is_file() else None,
        outcoupling_dir=outcoupling_dir if outcoupling_dir.is_dir() else None,
        light_stats=light_stats,
    )


def gpvdm_y_to_z_simulation_um(
    y_um: np.ndarray | float,
    *,
    total_thickness_um: float | None = None,
) -> np.ndarray | float:
    """Map gpvdm/Oghma epitaxy depth *y* (μm) to simulation TMM *z* (μm).

    gpvdm stacks layer0 at ``y=0`` (top / ITO anode) and increases *y* toward the
    cathode.  Simulation filmstack uses the same axis: ``z=0`` at the first finite
    epitaxy layer (TMM ``get_structure_pos`` layer index 1) and ``+z`` through
    the layer list; layer 0 is the semi-infinite ITO half-space at ``z→−∞``.

    For Oghma OLED projects this is the identity ``z = y``.  ``total_thickness_um``
    is accepted for API compatibility but unused in the unified convention.
    """
    del total_thickness_um
    y_arr = np.asarray(y_um, dtype=float)
    if np.ndim(y_um) == 0:
        return float(y_arr)
    return y_arr


def z_simulation_to_gpvdm_y_um(
    z_um: np.ndarray | float,
    *,
    total_thickness_um: float | None = None,
) -> np.ndarray | float:
    """Inverse of :func:`gpvdm_y_to_z_simulation_um` (identity for unified OLED z)."""
    return gpvdm_y_to_z_simulation_um(z_um, total_thickness_um=total_thickness_um)


DEFAULT_OUTCOUPLING_U_INTEGRATION_POINTS = 64
DEFAULT_RHO_U_MIN = 0.0
DEFAULT_RHO_U_MAX = 0.99


def outcoupling_u_integration_grid(
    n_u: int = DEFAULT_OUTCOUPLING_U_INTEGRATION_POINTS,
    *,
    u_min: float = 0.02,
    u_max: float = 0.99,
) -> tuple[np.ndarray, float]:
    """In-plane ``u`` samples and spacing for ``∫ P(u)·u·du`` (Oghma outcoupling-style quadrature)."""
    n_u = int(n_u)
    if n_u <= 1:
        return np.asarray([0.0], dtype=float), 1.0
    u_vals = np.linspace(float(u_min), float(u_max), n_u, dtype=float)
    return u_vals, float(u_vals[1] - u_vals[0])


def emission_rt_power_isotropic_at_z_py(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
    *,
    u_values: Sequence[float] | None = None,
    u: float = 0.0,
) -> tuple[float, float]:
    """Isotropic dipole R/T power at one depth; incoherently averages over ``u_values``."""
    if u_values is None:
        u_list = [float(u)]
    else:
        u_list = [float(ui) for ui in u_values]
    if not u_list:
        u_list = [0.0]
    p_top = 0.0
    p_bot = 0.0
    for ui in u_list:
        pt, pb = simulation_module.TMM_emission_solver_emission_rt_power_isotropic_at_z_s(
            layers, float(wl_um), complex(ui, 0.0), float(z_um)
        )
        p_top += float(pt)
        p_bot += float(pb)
    inv = 1.0 / len(u_list)
    return p_top * inv, p_bot * inv


def emission_rt_power_oriented_u_riemann_at_z_py(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
    *,
    orient: str = "hs",
    u_values: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Oriented R/T power at one depth with in-plane quadrature ``∫ P(u)·u·du``.

    Matches ``integrate_emission_r_t_power`` weighting in the C++ emission solver
  (incoherent sum of per-angle powers, not coherent ``E`` sum).
    """
    if u_values is None:
        u_vals, du = outcoupling_u_integration_grid()
    else:
        u_vals = np.asarray(u_values, dtype=float)
        du = float(u_vals[1] - u_vals[0]) if len(u_vals) > 1 else 1.0
    if len(u_vals) == 1 and float(u_vals[0]) == 0.0:
        return simulation_module.TMM_emission_solver_emission_rt_power_oriented_at_z_s(
            layers, float(wl_um), 0j, float(z_um), orient
        )
    fn = simulation_module.TMM_emission_solver_emission_rt_power_oriented_at_z_s
    p_top = 0.0
    p_bot = 0.0
    for ui in u_vals:
        weight = float(ui) * du
        pt, pb = fn(layers, float(wl_um), complex(float(ui), 0.0), float(z_um), orient)
        p_top += float(pt) * weight
        p_bot += float(pb) * weight
    return p_top, p_bot


def u_escape_halfspace(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    *,
    side: Literal["top", "bot"] = "top",
) -> float:
    """Propagating-mode cutoff ``u`` for escape into a semi-infinite bookend half-space.

    ``u = n·sinθ`` is conserved; radiative modes in the exit medium satisfy ``|u| ≤ Re(n_out)``.
    For top ITO bookend this is ``n_ITO·sin(π/2) = Re(n_ITO)``, not 1.
    """
    from coating_solver_test_util import material_nk_at_wl

    resolved = simulation_module.TMM_resolve_nk_depth_at_wl_s(layers, float(wl_um))
    bookend = resolved[0] if side == "top" else resolved[-1]
    n_out = material_nk_at_wl(bookend.background_material, float(wl_um))
    return float(abs(n_out.real))


def u_propagation_limit_at_z(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
) -> float:
    """Maximum propagating in-plane ``u`` in the layer hosting the dipole.

    ``u = n·sinθ`` with ``θ ≤ π/2`` in that layer gives ``u ≤ Re(n_layer(λ))``.
    This is the ``P_total`` angular upper limit (``n_EML·sin(π/2)`` when the source
    sits in the emissive layer).
    """
    from coating_solver_test_util import material_nk_at_wl

    resolved = simulation_module.TMM_resolve_nk_depth_at_wl_s(layers, float(wl_um))
    pos = simulation_module.TMM_get_structure_pos_s(resolved)
    layer_index, _ = simulation_module.TMM_find_in_structure_s(pos, float(z_um))
    layer = resolved[int(layer_index)]
    n_layer = material_nk_at_wl(layer.background_material, float(wl_um))
    return float(abs(n_layer.real))


def integrate_isotropic_p_top(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
    u_min: float,
    u_max: float,
    u_steps: int,
) -> float:
    """``∫_{u_min}^{u_max} p_top(u)·u·du`` for an isotropic dipole at ``z_um``."""
    p_top, _, _ = simulation_module.TMM_emission_solver_integrate_isotropic_power_at_z_s(
        layers,
        float(wl_um),
        float(z_um),
        float(u_min),
        float(u_max),
        int(u_steps),
    )
    return float(p_top)


def integrate_isotropic_p_top_escape_cone(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
    u_esc: float,
    u_min: float,
    u_steps_ref: int,
    u_max_ref: float,
) -> float:
    """``∫_{u_min}^{u_esc} p_top(u)·u·du`` with step count scaled to the escape interval."""
    u_esc = float(u_esc)
    u_max_ref = float(u_max_ref)
    steps_esc = _integration_steps_for_u_span(u_steps_ref, u_min, u_esc, u_max_ref)
    return integrate_isotropic_p_top(
        simulation_module, layers, wl_um, z_um, u_min, u_esc, steps_esc
    )


def _integration_steps_for_u_span(
    u_steps_ref: int,
    u_min: float,
    u_max: float,
    u_max_ref: float,
) -> int:
    u_steps_ref = int(u_steps_ref)
    u_span = float(u_max) - float(u_min)
    u_max_ref = float(u_max_ref)
    if u_steps_ref <= 0 or u_span <= 0.0:
        return max(10, u_steps_ref)
    if u_max_ref <= 0.0:
        return max(10, u_steps_ref)
    return max(10, int(u_steps_ref * u_span / u_max_ref))


def integrate_isotropic_k_diss(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
    u_min: float,
    u_max: float,
    u_steps: int,
) -> float:
    """``∫_{u_min}^{u_max} k_diss(u)·u·du`` — total isotropic dissipation (LDOS proxy)."""
    _, _, k_diss = simulation_module.TMM_emission_solver_integrate_isotropic_power_at_z_s(
        layers,
        float(wl_um),
        float(z_um),
        float(u_min),
        float(u_max),
        int(u_steps),
    )
    return float(k_diss)


def isotropic_outcoupling_fraction_at_z(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float,
    z_um: float,
    u_esc: float,
    u_min: float,
    u_steps_ref: int,
    u_max_ref: float,
) -> float:
    """Escape fraction ``η = P_escape / P_total`` at one depth.

    ``P_escape = ∫_{u_min}^{u_esc} p_top·u·du`` — outcoupled power within the exit-medium
    escape cone (``u_esc = Re(n_exit)·sin(π/2)`` from :func:`u_escape_halfspace`).

    ``P_total  = ∫_{u_min}^{u_eml} k_diss·u·du`` — total dipole dissipation over all
    propagating modes in the source layer (``u_eml = Re(n_EML)·sin(π/2)`` from
    :func:`u_propagation_limit_at_z`).

    The two integrals use **different upper limits** when ``n_exit ≠ n_EML``.
    """
    u_esc = float(u_esc)
    u_eml = u_propagation_limit_at_z(simulation_module, layers, wl_um, z_um)
    u_escape_hi = max(u_esc, float(u_min))
    u_total_hi = max(u_eml, float(u_min))
    if u_escape_hi <= float(u_min) or u_total_hi <= float(u_min):
        return 0.0

    steps_escape = _integration_steps_for_u_span(u_steps_ref, u_min, u_escape_hi, u_max_ref)
    steps_total = _integration_steps_for_u_span(u_steps_ref, u_min, u_total_hi, u_max_ref)
    p_escape = integrate_isotropic_p_top(
        simulation_module, layers, wl_um, z_um, u_min, u_escape_hi, steps_escape
    )
    p_total = integrate_isotropic_k_diss(
        simulation_module, layers, wl_um, z_um, u_min, u_total_hi, steps_total
    )
    if p_total < 1e-30:
        return 0.0
    return float(min(p_escape / p_total, 1.0))


def tmm_stack_z_edges_from_layer_depths(
    layer_depths_nm: Sequence[float],
    *,
    entry_bookend_below_z0: bool = True,
) -> list[float]:
    """Return z interface positions matching ``TMM_helper::get_structure_pos``.

    Parameter name ``layer_depths_nm`` follows ``coating_visualizer`` (nm when
    called from ``plot_coating_stack``, which scales μm×1000). Oghma call sites pass
    μm layer depths directly; values are accumulated in the same unit as TMM
    ``layer.depth``.
    """
    import coating_visualizer as fv

    return fv.tmm_stack_z_edges_from_layer_depths(
        layer_depths_nm, entry_bookend_below_z0=entry_bookend_below_z0
    )


def tmm_stack_z_edges_from_bar_thicknesses(
    thicknesses_um: Sequence[float],
    *,
    entry_bookend_below_z0: bool = False,
) -> tuple[list[float], list[float]]:
    """Parse bar-chart thicknesses ``[0, *layer_depths*, 0]`` (μm) into z edges.

    Forwards to ``coating_visualizer`` (internal param names use nm when the
    filmstack plot path scales μm×1000). This wrapper expects **μm** depths.
    """
    import coating_visualizer as fv

    return fv.tmm_stack_z_edges_from_bar_thicknesses(
        thicknesses_um, entry_bookend_below_z0=entry_bookend_below_z0
    )



def _make_oghma_optical_material(
    simulation_module: Any,
    material_path: str,
) -> Any:
    del simulation_module
    return _read_oghma_material(material_path)


def _structure_pos_from_layers(layers: list[Any]) -> list[float]:
    pos = [float("-inf"), 0.0]
    for lyr in layers[1:]:
        pos.append(pos[-1] + float(lyr.depth))
    return pos




def vacuum_reference_layers(simulation_module: Any, lam_um: float, thickness_um: float = 2.0):
    """Homogeneous vacuum slab for outcoupling enhancement reference."""
    del lam_um
    from coating_solver_test_util import make_coating

    air_nk = 1.0 + 0.0j
    return [
        make_coating(simulation_module, air_nk, 0.0, "air"),
        make_coating(simulation_module, air_nk, float(thickness_um), "air"),
    ]









def normalize_rows(arr: np.ndarray, axis: int = 1, floor: float = 1e-30) -> np.ndarray:
    """Normalize each row by its maximum (for depth profiles at fixed λ)."""
    peak = np.maximum(np.max(arr, axis=axis, keepdims=True), floor)
    return arr / peak


def plot_2d_compare(
    wl: np.ndarray,
    y: np.ndarray,
    tmm_grid: np.ndarray,
    baseline_grid: np.ndarray,
    title: str,
    *,
    row_norm: bool = True,
    out_path: Path | str | None = None,
    baseline_label: str = "baseline",
    ours_label: str = "TMM",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-panel (λ, y) heatmap: ours, baseline, |error| — same layout as ``02_oled_emission_tmm``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tg = normalize_rows(tmm_grid.copy()) if row_norm else tmm_grid.copy()
    bg = normalize_rows(baseline_grid.copy()) if row_norm else baseline_grid.copy()
    err = np.abs(tg - bg)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, lab in zip(axes[:2], [tg, bg], [ours_label, baseline_label]):
        im = ax.pcolormesh(wl, y, data.T, shading="auto", cmap="viridis")
        ax.set_xlabel("λ (μm)")
        ax.set_ylabel("y (μm)")
        ax.set_title(lab)
        cbar_label = "row-norm" if row_norm else "value"
        plt.colorbar(im, ax=ax, label=cbar_label)
    im3 = axes[2].pcolormesh(wl, y, err.T, shading="auto", cmap="magma")
    axes[2].set_xlabel("λ (μm)")
    axes[2].set_ylabel("y (μm)")
    axes[2].set_title("|error|")
    plt.colorbar(im3, ax=axes[2], label="|error|")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return tg, bg, err




def optical_output_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / "optical_output"


def filter_stack_layers(project: OghmaProject) -> list[OghmaLayer]:
    """Return epitaxy layers excluding the finite air incident region."""
    return [layer for layer in project.layers if layer.name.lower() != "air"]


def _layer_role_tag(layer: OghmaLayer) -> str:
    tag = layer.obj_type
    if layer.pl_emission_enabled:
        tag = f"{tag},EML"
    return tag



def format_oghma_epitaxy_roles(project: OghmaProject) -> str:
    """One-line Oghma epitaxy with Layer editor types (bottom→top, y=0 @ ITO)."""
    labels = [
        f"{layer.name.split()[0]}({layer.thickness_um:g},{_layer_role_tag(layer)})"
        for layer in project.layers
    ]
    return " / ".join(labels)




def air_layer_thickness_um(project: OghmaProject) -> float:
    for layer in project.layers:
        if layer.name.lower() == "air":
            return float(layer.thickness_um)
    return 0.0



def _heatmap_grid_from_series(
    wl: np.ndarray,
    y_um: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    """Build (λ, y) grid from sparse heatmap CSV columns."""
    wl_unique = np.unique(wl)
    y_unique = np.unique(y_um)
    grid = np.full((len(wl_unique), len(y_unique)), np.nan)
    wl_to_i = {w: i for i, w in enumerate(wl_unique)}
    y_to_i = {y: i for i, y in enumerate(y_unique)}
    for w, y, v in zip(wl, y_um, values):
        grid[wl_to_i[w], y_to_i[y]] = v
    return {"heat_wl_um": wl_unique, "heat_y_um": y_unique, "grid": grid}


def _heatmap_from_csv(path: Path) -> dict[str, Any]:
    heat = parse_oghma_csv(path)
    xs = _oghma_axis_to_um(heat["x"], heat["meta"], heatmap=True)
    ys = _oghma_axis_to_um(heat["y"], heat["meta"], heatmap=True)
    grid_data = _heatmap_grid_from_series(xs, ys, heat["z"])
    return {
        "heat_wl_um": grid_data["heat_wl_um"],
        "heat_y_um": grid_data["heat_y_um"],
        "grid": grid_data["grid"],
    }


def load_oghma_optical_reference(project: OghmaProject | Path | str) -> dict[str, Any]:
    """Load reflect/transmit spectra and photon heatmaps from optical_output/."""
    if not isinstance(project, OghmaProject):
        project = load_oghma_project(project)
    out_dir = optical_output_dir(project.project_dir)
    out: dict[str, Any] = {"project_dir": project.project_dir}

    for key, fname in (
        ("reflect", "reflect.csv"),
        ("transmit", "transmit.csv"),
    ):
        path = out_dir / fname
        if path.is_file():
            d = parse_oghma_csv(path)
            out[f"{key}_wl_um"] = _oghma_axis_to_um(d["x"], d["meta"], axis="y")
            out[f"{key}"] = d["y"]

    for key, fname in (
        ("photons", "photons_yl.csv"),
        ("photons_abs", "photons_abs_yl.csv"),
    ):
        path = out_dir / fname
        if path.is_file():
            heat = _heatmap_from_csv(path)
            out[f"{key}_wl_um"] = heat["heat_wl_um"]
            out[f"{key}_y_um"] = heat["heat_y_um"]
            out[f"{key}_grid"] = heat["grid"]

    if project.light_stats:
        out["light_stats"] = dict(project.light_stats)
    return out


def list_oghma_optical_snapshots(project_dir: Path | str) -> list[dict[str, Any]]:
    """List wavelength snapshots under optical_snapshots/."""

    def _parse(meta: dict[str, Any]) -> dict[str, Any]:
        return {"wl_um": float(meta.get("wavelength", np.nan)) * 1e6}

    return _list_indexed_snapshots(project_dir, "optical_snapshots", parse_meta=_parse)



def load_oghma_optical_snapshot(project_dir: Path | str, index: int) -> dict[str, Any]:
    """Load photons.csv depth profile from one optical snapshot."""

    def _load(snap_dir: Path, out: dict[str, Any]) -> None:
        if "meta" in out:
            out["wl_um"] = float(out["meta"].get("wavelength", np.nan)) * 1e6
        path = snap_dir / "photons.csv"
        if path.is_file():
            d = parse_oghma_csv(path)
            out["y_um"] = _oghma_axis_to_um(d["x"], d["meta"], heatmap=True)
            out["photons"] = d["y"]

    return _load_indexed_snapshot(project_dir, "optical_snapshots", index, load_files=_load)



def eml_z_bounds_um(project: OghmaProject) -> tuple[float, float, float]:
    """Return (z_lo, z_hi, z_center) for the EML in filmstack *z* (``z = y_oghma``)."""
    eml = project.eml_layer()
    if eml is None:
        raise ValueError("no emissive layer (pl_emission_enabled) in project")
    z0 = float(eml.y0_um)
    z1 = float(eml.y1_um)
    z_lo = min(z0, z1)
    z_hi = max(z0, z1)
    return z_lo, z_hi, 0.5 * (z_lo + z_hi)



def build_oghma_passive_stack(
    project: OghmaProject,
    lam_um: float,
    simulation_module: Any,
    *,
    exit_halfspace: str = "air",
    substrate_depth_nm: float = SUBSTRATE_DEPTH_UM,
):
    """
    Build passive TMM stack for optical-filter projects (bottom incidence).

    Layer order (TMM z from 0 at device bottom): semi-inf air | air(100 nm) |
    layer0 | … | layerN | semi-inf exit.  ``exit_halfspace`` is ``"air"`` (01) or
    ``"top_layer"`` (02, last epitaxy material).
    """
    del substrate_depth_nm
    return _tmm_layers_from_project_formula(
        passive_filter_formula_from_project(project, exit_halfspace=exit_halfspace),
        passive_filter_materials_db(project, exit_halfspace=exit_halfspace),
        lam_um,
        simulation_module,
    )


def _filter_layer_token(layer: OghmaLayer) -> str:
    return layer.name.split()[0].replace(":", "")


def passive_filter_formula_from_project(
    project: OghmaProject,
    *,
    exit_halfspace: str = "air",
) -> str:
    """Filmstack formula aligned with :func:`build_oghma_passive_stack`."""
    stack = filter_stack_layers(project)
    if not stack:
        raise ValueError("optical filter project has no epitaxy layers")
    air_um = float(air_layer_thickness_um(project))
    parts = ["Air 0", f"air {air_um:.5f}"]
    for layer in stack:
        parts.append(f"{_filter_layer_token(layer)} {layer.thickness_um:.5f}")
    if exit_halfspace == "top_layer":
        parts.append(f"{_filter_layer_token(stack[-1])} 0")
    else:
        parts.append("ExitLayer 0")
    return " ".join(parts)


def passive_filter_materials_db(
    project: OghmaProject,
    *,
    exit_halfspace: str = "air",
) -> dict[str, Any]:
    """Materials map for :func:`passive_filter_formula_from_project`."""
    stack = filter_stack_layers(project)
    db: dict[str, Any] = {}
    for layer in stack:
        db[_filter_layer_token(layer)] = _read_oghma_material(layer.optical_material)
    if exit_halfspace != "top_layer":
        db["ExitLayer"] = _read_oghma_material("generic/n/1.0")
    return db


def passive_filter_stack_labels(project: OghmaProject) -> list[str]:
    """Finite-layer labels for :func:`build_oghma_passive_stack` (incident → exit)."""
    stack = filter_stack_layers(project)
    labels = ["air"]
    for layer in stack:
        mat = layer.optical_material.split("/")[-1]
        labels.append(f"{layer.name.split()[0]}(n={mat})")
    return labels


def _spectrum_y_map(
    wl_um: np.ndarray,
    y_um: np.ndarray,
    profile_at_wl: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    out = np.zeros((len(wl_um), len(y_um)))
    for i, wl in enumerate(wl_um):
        out[i] = profile_at_wl(float(wl), y_um)
    return out


def _layers_from_builder(
    build_layers: Callable[[float], list[Any]] | list[Any],
    wl_um: float,
) -> list[Any]:
    if callable(build_layers):
        return build_layers(float(wl_um))
    return build_layers


def _list_indexed_snapshots(
    project_dir: Path | str,
    subdir: str,
    *,
    parse_meta: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    require_meta: bool = False,
) -> list[dict[str, Any]]:
    project_dir = Path(project_dir)
    snap_dir = project_dir / subdir
    if not snap_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for sub in sorted(snap_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        meta_path = sub / "data.json"
        if require_meta and not meta_path.is_file():
            continue
        extra: dict[str, Any] = {}
        if meta_path.is_file() and parse_meta is not None:
            extra = parse_meta(json.loads(meta_path.read_text()))
        entries.append({"index": int(sub.name), "path": sub, **extra})
    return entries


def _load_indexed_snapshot(
    project_dir: Path | str,
    subdir: str,
    index: int,
    *,
    load_files: Callable[[Path, dict[str, Any]], None],
) -> dict[str, Any]:
    project_dir = Path(project_dir)
    snap_dir = project_dir / subdir / str(index)
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"snapshot not found: {snap_dir}")
    out: dict[str, Any] = {"index": index, "path": snap_dir}
    meta_path = snap_dir / "data.json"
    if meta_path.is_file():
        out["meta"] = json.loads(meta_path.read_text())
    load_files(snap_dir, out)
    return out


def compare_arrays(ours: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, max abs error, and Pearson correlation."""
    a = np.asarray(ours, dtype=float)
    b = np.asarray(baseline, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"rmse": np.nan, "mae": np.nan, "max_abs": np.nan, "corr": np.nan}
    diff = a[mask] - b[mask]
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "corr": float(np.corrcoef(a[mask], b[mask])[0, 1]),
    }


def compare_stop_band_metrics(
    ours: np.ndarray,
    baseline: np.ndarray,
    wl_um: np.ndarray,
    *,
    notch_fraction: float = 0.1,
) -> dict[str, float]:
    """Stop-band notch alignment: argmin wavelength delta and log-RMSE in notch region.

    Returned keys ``stop_x_ours``, ``stop_x_baseline``, and ``stop_delta_um`` are in μm.
    """
    ours = np.asarray(ours, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    wl = np.asarray(wl_um, dtype=float)
    i_ours = int(np.argmin(ours))
    i_base = int(np.argmin(baseline))
    thresh = notch_fraction * max(float(np.max(baseline)), 1e-30)
    mask = baseline < thresh
    log_ours = np.log10(np.maximum(ours, 1e-30))
    log_base = np.log10(np.maximum(baseline, 1e-30))
    out: dict[str, float] = {
        "stop_x_ours": float(wl[i_ours]),
        "stop_x_baseline": float(wl[i_base]),
        "stop_delta_um": abs(float(wl[i_ours] - wl[i_base])),
    }
    if np.any(mask):
        out["stop_log_rmse"] = float(
            np.sqrt(np.mean((log_ours[mask] - log_base[mask]) ** 2))
        )
    return out


def peak_normalize_spectrum(values: np.ndarray) -> np.ndarray:
    """Peak-normalize a 1D spectrum by its own maximum (dimensionless shape comparison)."""
    arr = np.asarray(values, dtype=float)
    peak = max(float(np.max(arr)), 1e-30)
    return arr / peak


def compare_metrics(
    ours: np.ndarray,
    baseline: np.ndarray,
    *,
    x: np.ndarray | None = None,
    peak_axis: int | None = None,
) -> dict[str, float]:
    """RMSE, MAE, max abs error, correlation; optional peak position delta on axis."""
    stats = compare_arrays(ours, baseline)
    if x is not None and peak_axis is not None:
        a = np.asarray(ours, dtype=float)
        b = np.asarray(baseline, dtype=float)
        if a.ndim == 1:
            stats["peak_x_ours"] = float(x[int(np.argmax(a))])
            stats["peak_x_baseline"] = float(x[int(np.argmax(b))])
            stats["peak_delta"] = abs(stats["peak_x_ours"] - stats["peak_x_baseline"])
    return stats


def _cplx_from_py(z: Any) -> complex:
    import tmm_utils

    return tmm_utils.cplx_from_py(z)


def spectrum_reflection_coeff_unpolarized_s(
    simulation_module: Any,
    layers: list[Any],
    wl_um: float | np.ndarray,
    *,
    incident_angle_rad: float = 0.0,
) -> np.ndarray:
    """Batch unpolarized reflection coefficient r(λ) via C++ spectrum API."""
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    angle_c = complex(float(incident_angle_rad), 0.0)
    r_list = simulation_module.TMM_solver_spectrum_reflection_coeff_unpolarized_s(
        layers, list(wl_arr.astype(float)), angle_c
    )
    return np.asarray([_cplx_from_py(r) for r in r_list], dtype=complex)


def _emission_e_out_at_z_py(
    simulation_module: Any,
    layers: list[Any],
    wl: float,
    z_um: float,
    orient: str,
    spectrum_amp: float,
    u: float = 0.0,
    n_u: int = 1,
    u_span: float | None = None,
) -> tuple[complex, complex]:
    """Coherent E_out_top/bot at one wavelength (resolved runtime stack)."""
    resolved = simulation_module.TMM_resolve_nk_depth_at_wl_s(layers, float(wl))
    solve_fn = _emission_solve_fn(simulation_module, orient)
    amp = math.sqrt(max(float(spectrum_amp), 0.0))
    n_u = int(n_u)
    if n_u > 1:
        span = float(u_span if u_span is not None else 0.98 - max(u, 0.02))
        u_start = max(float(u) - span / 2.0, 0.02)
        u_end = min(float(u) + span / 2.0, 0.99)
        du = (u_end - u_start) / (n_u - 1)
        u_vals = [u_start + i * du for i in range(n_u)]
    else:
        u_vals = [float(u)]
    e_top = complex(0.0)
    e_bot = complex(0.0)
    for ui in u_vals:
        result = solve_fn(resolved, float(wl), complex(ui, 0.0), float(z_um))
        e_top += _cplx_from_py(result.E_out_top) * amp
        e_bot += _cplx_from_py(result.E_out_bot) * amp
    return e_top, e_bot


def fdtd_input_combine_power(
    r_coeff: np.ndarray,
    *,
    eps: float = 1e-30,
) -> np.ndarray:
    """Input-plane power factor ``G_in = |1 + r|²`` (incident + reflected coherent sum)."""
    r = np.asarray(r_coeff, dtype=complex)
    amp = np.maximum(np.abs(1.0 + r), math.sqrt(eps))
    return np.maximum(amp**2, eps)


def lam_e_norm_from_transfer(
    t_passive: np.ndarray,
    r_coeff: np.ndarray,
) -> np.ndarray:
    """FDTD ``lam_E_norm`` model: ``T_passive(λ) / G_in(λ)`` with ``G_in = |1 + r|²``."""
    return np.asarray(t_passive, dtype=float) / fdtd_input_combine_power(r_coeff)


def build_emission_light_source(
    simulation_module: Any,
    lam_um: float,
    spectrum_amp: float,
    *,
    u: float = 0.0,
    n_u: int = 1,
    u_span: float | None = None,
) -> Any:
    """Build ``emission_light_source_s`` for a single wavelength with coherent amplitude."""
    n_u = int(n_u)
    if n_u < 1:
        raise ValueError("n_u must be positive")

    src = simulation_module.emission_light_source_s()
    src.shape = [1, n_u]
    if n_u > 1:
        span = float(u_span if u_span is not None else 0.98 - max(u, 0.02))
        u_start = max(float(u) - span / 2.0, 0.02)
        u_end = min(float(u) + span / 2.0, 0.99)
        du = (u_end - u_start) / (n_u - 1)
    else:
        u_start = float(u)
        du = 0.0
    src.grid.start = [float(lam_um), u_start]
    src.grid.step = [0.0, du]
    amp = math.sqrt(max(float(spectrum_amp), 0.0))
    src.data = [amp] * n_u
    return src


def _emission_source_shape(source: Any) -> tuple[int, int]:
    shape = list(source.shape)
    return int(shape[0]), int(shape[1])


def _emission_solve_fn(simulation_module: Any, orient: str) -> Any:
    fn_map = {
        "hs": simulation_module.TMM_emission_solve_hs_s,
        "hp": simulation_module.TMM_emission_solve_hp_s,
        "vp": simulation_module.TMM_emission_solve_vp_s,
    }
    if orient not in fn_map:
        raise ValueError(f"unknown orient: {orient!r}")
    return fn_map[orient]


def _compute_emission_transfer(
    layout: Any,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    build_stack_at_wl: Callable[[float], list[Any]] | list[Any],
    reflection_at_z: Callable[[float, np.ndarray], np.ndarray],
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
    z_g_in_nm: float | None = None,
) -> dict[str, np.ndarray]:
    if z_um is None:
        z_um = float(layout.z_source_um)
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    if callable(build_stack_at_wl) and wl_arr.size > 1:
        chunks = [
            _compute_emission_transfer(
                layout,
                float(wl),
                simulation_module,
                build_stack_at_wl=build_stack_at_wl,
                reflection_at_z=reflection_at_z,
                z_um=z_um,
                u=u,
                orient=orient,
                z_g_in_nm=z_g_in_nm,
            )
            for wl in wl_arr
        ]
        return {
            key: np.concatenate([np.atleast_1d(chunk[key]) for chunk in chunks])
            for key in chunks[0]
        }

    layers = _layers_from_builder(build_stack_at_wl, float(wl_arr[0]))
    z_g_in = float(z_g_in_nm if z_g_in_nm is not None else layout.z_detector_in_um)
    r_coeff = np.asarray(reflection_at_z(z_g_in, wl_arr), dtype=complex)
    p_top, p_bot, g_in, _r, t_ratio, t_trans_ratio, t_trans_ratio_fdtd = (
        simulation_module.TMM_emission_solver_spectrum_emission_transfer_s(
            layers,
            wl_arr.tolist(),
            float(z_um),
            float(u),
            orient,
            [complex(v) for v in r_coeff],
        )
    )
    return {
        "p_top": np.asarray(p_top, dtype=float),
        "p_bot": np.asarray(p_bot, dtype=float),
        "g_in": np.asarray(g_in, dtype=float),
        "r_coeff": r_coeff,
        "t_ratio": np.asarray(t_ratio, dtype=float),
        "t_trans_ratio": np.asarray(t_trans_ratio, dtype=float),
        "t_trans_ratio_fdtd": np.asarray(t_trans_ratio_fdtd, dtype=float),
    }


def _compute_emission_coherent_fields_at_z(
    layout: Any,
    wl_um: float | np.ndarray,
    simulation_module: Any,
    *,
    build_stack_at_wl: Callable[[float], list[Any]] | list[Any],
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
    spectrum: np.ndarray | None = None,
    n_u: int = 1,
    u_span: float | None = None,
) -> dict[str, np.ndarray]:
    if z_um is None:
        z_um = float(layout.z_source_um)
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    if spectrum is None:
        spectrum_arr = np.ones_like(wl_arr, dtype=float)
    else:
        spectrum_arr = np.asarray(spectrum, dtype=float)
        if spectrum_arr.shape != wl_arr.shape:
            raise ValueError(
                f"spectrum shape must match wl_um: {spectrum_arr.shape} vs {wl_arr.shape}"
            )

    e_top_list: list[complex] = []
    e_bot_list: list[complex] = []
    for i, wl in enumerate(wl_arr):
        layers = _layers_from_builder(build_stack_at_wl, float(wl))
        e_top, e_bot = _emission_e_out_at_z_py(
            simulation_module,
            layers,
            float(wl),
            float(z_um),
            orient,
            float(spectrum_arr[i]),
            u=u,
            n_u=n_u,
            u_span=u_span,
        )
        e_top_list.append(e_top)
        e_bot_list.append(e_bot)

    return {
        "e_top": np.asarray(e_top_list, dtype=complex),
        "e_bot": np.asarray(e_bot_list, dtype=complex),
    }


def _predict_coupled_spectrum_coherent_p_bot(
    simulation_module: Any,
    wl_um: float | np.ndarray,
    e_bot: np.ndarray,
    *,
    build_stack_at_wl: Callable[[float], list[Any]] | list[Any],
    u: float = 0.0,
) -> np.ndarray:
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    e_bot_arr = np.asarray(e_bot, dtype=complex)
    if e_bot_arr.shape != wl_arr.shape:
        raise ValueError(
            f"e_bot shape must match wl_um: {e_bot_arr.shape} vs {wl_arr.shape}"
        )
    layers = _layers_from_builder(build_stack_at_wl, float(wl_arr[0]))
    flux_list: list[float] = []
    for i, wl in enumerate(wl_arr):
        flux = simulation_module.TMM_emission_solver_emission_boundary_flux_at_wl_s(
            layers,
            float(wl),
            float(u),
            complex(e_bot_arr[i]),
            "bot",
        )
        flux_list.append(float(flux))
    return np.asarray(flux_list, dtype=float)


def _compute_emission_case_bc_spectra(
    simulation_module: Any,
    layout: Any,
    wl_um: float | np.ndarray,
    s_in: np.ndarray,
    *,
    build_stack_at_wl: Callable[[float], list[Any]],
    compute_passive_transfer_fn: Callable[[np.ndarray], np.ndarray],
    predict_case_c_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    spectrum: np.ndarray | None = None,
    t_passive: np.ndarray | None = None,
    z_um: float | None = None,
    u: float = 0.0,
    orient: str = "hs",
) -> dict[str, np.ndarray]:
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    s_in = np.asarray(s_in, dtype=float)
    if spectrum is None:
        spectrum = s_in
    source_spectrum = np.asarray(spectrum, dtype=float)
    if t_passive is None:
        t_passive = compute_passive_transfer_fn(wl_arr)
    coh = _compute_emission_coherent_fields_at_z(
        layout,
        wl_arr,
        simulation_module,
        build_stack_at_wl=build_stack_at_wl,
        z_um=z_um,
        u=u,
        orient=orient,
        spectrum=source_spectrum,
    )
    s_pred_b = _predict_coupled_spectrum_coherent_p_bot(
        simulation_module,
        wl_arr,
        coh["e_bot"],
        build_stack_at_wl=build_stack_at_wl,
        u=u,
    )
    s_pred_c = predict_case_c_fn(source_spectrum, np.asarray(t_passive, dtype=float))
    return {
        "s_pred_b": s_pred_b,
        "s_pred_c": s_pred_c,
        "t_passive": np.asarray(t_passive, dtype=float),
        **coh,
    }







