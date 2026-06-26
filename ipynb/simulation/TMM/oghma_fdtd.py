"""Shared Oghma FDTD layout, source, and detector reference parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from oghma_core import _oghma_axis_to_um, parse_oghma_csv

_DEFAULT_REQUIRED_DETECTOR_FILES: tuple[str, ...] = (
    "detector0/lam_E.csv",
    "detector1/lam_E.csv",
)


@dataclass(frozen=True)
class FdtdSourceParams:
    """Oghma FDTD soft-source parameters (``gaus_sin`` etc.)."""

    waveform: str
    amp: float
    t0_fs: float
    tau_fs: float
    excite_ey: bool


@dataclass(frozen=True)
class FdtdLayout:
    """FDTD source, detector, and replicated-stack z positions (μm, bottom = z=0)."""

    z_source_um: float
    z_detector_in_um: float
    z_detector_out_um: float
    z_stack_start_um: float


def _oghma_length_m_to_um(value_m: float) -> float:
    """Convert a length in metres to micrometres."""
    return float(value_m) * 1e6


def _time_to_fs(values: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Convert Oghma detector ``power.csv`` time column to femtoseconds."""
    _ = meta
    return np.asarray(values, dtype=float) * 1e15


def _load_sim_dict(sim_path: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(sim_path, dict):
        return sim_path
    return json.loads(Path(sim_path).read_text())


def _parse_world_segment0(sim: dict[str, Any]) -> dict[str, Any]:
    world = sim.get("world", {})
    world_data = world.get("world_data", world)
    n_seg = int(world_data.get("segments", 0))
    if n_seg < 1:
        raise ValueError("sim.json has no world segments")
    return world_data["segment0"]


def parse_fdtd_layout(sim_path: Path | str | dict[str, Any]) -> FdtdLayout:
    """Parse FDTD source, detector, and stack z positions from ``sim.json``."""
    sim = _load_sim_dict(sim_path)
    src_seg = sim["optical"]["light_sources"]["lights"]["segment0"]
    det = sim["optical"]["detectors"]
    stack_seg = _parse_world_segment0(sim)
    return FdtdLayout(
        z_source_um=_oghma_length_m_to_um(float(src_seg["z0"])),
        z_detector_in_um=_oghma_length_m_to_um(float(det["segment0"]["z0"])),
        z_detector_out_um=_oghma_length_m_to_um(float(det["segment1"]["z0"])),
        z_stack_start_um=_oghma_length_m_to_um(float(stack_seg["z0"])),
    )


def parse_fdtd_source(sim_path: Path | str | dict[str, Any]) -> FdtdSourceParams:
    """Parse FDTD source settings from ``sim.json`` ``light_sources``."""
    sim = _load_sim_dict(sim_path)
    ls = sim["optical"]["light_sources"]["lights"]["segment0"]["light_source"]["fdtd"]

    def _to_fs(key: str) -> float:
        val = float(ls.get(key, 0.0))
        unit = str(ls.get(f"{key}_u", "s")).lower()
        if unit == "fs":
            return val
        if unit == "s":
            return val * 1e15
        return val * 1e15

    return FdtdSourceParams(
        waveform=str(ls.get("fdtd_src_waveform", "")),
        amp=float(ls.get("fdtd_src_amp", 1.0)),
        t0_fs=_to_fs("fdtd_src_t0_time"),
        tau_fs=_to_fs("fdtd_src_tau_time"),
        excite_ey=str(ls.get("fdtd_excite_Ey", "True")).lower() == "true",
    )


def require_fdtd_reference_keys(ref: dict[str, Any], *keys: str) -> None:
    """Raise if expected REF keys are absent after loading."""
    missing = [k for k in keys if k not in ref]
    if missing:
        raise KeyError(f"REF missing keys {missing}; loaded keys: {sorted(ref)}")


def load_oghma_fdtd_detector_reference(
    project_dir: Path | str,
    *,
    required: Sequence[str] | None = _DEFAULT_REQUIRED_DETECTOR_FILES,
) -> dict[str, Any]:
    """Load Oghma FDTD detector spectra and time-domain power from ``detector0/1/``."""
    project_dir = Path(project_dir)
    if required:
        missing = [rel for rel in required if not (project_dir / rel).is_file()]
        if missing:
            paths = "\n".join(str(project_dir / rel) for rel in missing)
            raise FileNotFoundError(
                f"Missing required Oghma FDTD detector files under {project_dir}:\n{paths}"
            )

    out: dict[str, Any] = {"project_dir": project_dir}

    for det_key, det_name in (("input", "detector0"), ("output", "detector1")):
        det_dir = project_dir / det_name
        for fname, suffix in (
            ("lam_E.csv", "lam_e"),
            ("power.csv", "power"),
        ):
            path = det_dir / fname
            if not path.is_file():
                continue
            d = parse_oghma_csv(path)
            meta = d["meta"]
            if suffix == "lam_e":
                out[f"{det_key}_{suffix}_wl_um"] = _oghma_axis_to_um(d["x"], meta, axis="y")
                out[f"{det_key}_{suffix}"] = d["y"]
            else:
                out[f"{det_key}_{suffix}_t_fs"] = _time_to_fs(d["x"], meta)
                out[f"{det_key}_{suffix}"] = d["y"]

    norm_path = project_dir / "detector1" / "lam_E_norm.csv"
    if norm_path.is_file():
        d = parse_oghma_csv(norm_path)
        out["transmit_norm_wl_um"] = _oghma_axis_to_um(d["x"], d["meta"], axis="y")
        out["transmit_norm"] = d["y"]

    return out
