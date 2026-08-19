"""FDTD waveform → TMM incoherent source spectrum (I_new = |E(f)|² × Intensity(f))."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from oghma_core import _cplx_from_py, _read_oghma_spectrum
from oghma_fdtd import _oghma_length_m_to_um

_C_LIGHT_UM_PER_S = 299_792_458.0 * 1e6
_C_LIGHT_M_PER_S = 299_792_458.0
_FS_TO_S = 1e-15


@dataclass(frozen=True)
class FdtdLightSourceConfig:
    """Oghma FDTD light source: waveform + spectral envelope."""

    waveform: str
    amp: float
    phase_rad: float
    dc_offset: float
    t0_fs: float
    tau_fs: float
    t_start_fs: float
    t_end_fs: float
    window: str
    window_alpha: float
    light_spectrum: str
    light_multiplier: float
    carrier_wl_um: float
    excite_ey: bool


def _to_fs(val: float, unit: str) -> float:
    u = str(unit).lower()
    fval = float(val)
    if u == "s":
        return fval * 1e15
    if u == "fs":
        # Oghma often stores SI seconds while ``*_u`` reads ``fs`` (values ≪ 1).
        if 0 < abs(fval) < 1.0:
            return fval * 1e15
        return fval
    return fval * 1e15


def _parse_carrier_wl_um(sim: dict[str, Any]) -> float:
    optical = sim.get("optical", {})
    mesh = optical.get("mesh", {}).get("mesh_l", {})
    seg = mesh.get("segment0")
    if isinstance(seg, dict):
        start = _oghma_length_m_to_um(float(seg.get("start", 0.0)))
        stop = _oghma_length_m_to_um(float(seg.get("stop", start)))
        if stop > start:
            return 0.5 * (start + stop)
        if start > 0:
            return start
    lasers = optical.get("lasers", {})
    lseg = lasers.get("segment0", {}).get("config", {})
    wl_m = float(lseg.get("laserwavelength", 0.0))
    if wl_m > 0:
        return wl_m * 1e6
    return 0.5


def parse_fdtd_light_source_config(sim_path: Path | str | dict[str, Any]) -> FdtdLightSourceConfig:
    """Parse FDTD waveform + ``light_spectrum`` from ``sim.json``."""
    if isinstance(sim_path, dict):
        sim = sim_path
    else:
        sim = json.loads(Path(sim_path).read_text())

    src = sim["optical"]["light_sources"]["lights"]["segment0"]["light_source"]
    fdtd = src["fdtd"]
    spectra = src.get("light_spectra", {})
    spec_seg = spectra.get("segment0", {}) if int(spectra.get("segments", 0)) >= 1 else {}

    def _fdtd_fs(key: str) -> float:
        return _to_fs(float(fdtd.get(key, 0.0)), str(fdtd.get(f"{key}_u", "s")))

    phase_deg = float(fdtd.get("fdtd_src_phase_deg", 0.0))
    return FdtdLightSourceConfig(
        waveform=str(fdtd.get("fdtd_src_waveform", "")),
        amp=float(fdtd.get("fdtd_src_amp", 1.0)),
        phase_rad=math.radians(phase_deg),
        dc_offset=float(fdtd.get("fdtd_src_dc_offset", 0.0)),
        t0_fs=_fdtd_fs("fdtd_src_t0_time"),
        tau_fs=_fdtd_fs("fdtd_src_tau_time"),
        t_start_fs=_fdtd_fs("fdtd_src_start_time"),
        t_end_fs=_fdtd_fs("fdtd_src_end_time"),
        window=str(fdtd.get("fdtd_src_window", "none")).lower(),
        window_alpha=float(fdtd.get("fdtd_src_window_alpha", 0.2)),
        light_spectrum=str(spec_seg.get("light_spectrum", "AM1.5G")),
        light_multiplier=float(spec_seg.get("light_multiplyer", 1.0)),
        carrier_wl_um=_parse_carrier_wl_um(sim),
        excite_ey=str(fdtd.get("fdtd_excite_Ey", "True")).lower() == "true",
    )


def load_oghma_light_spectrum(
    name: str,
    wl_um: np.ndarray,
    *,
    multiplier: float = 1.0,
) -> np.ndarray:
    """Interpolate ``light_spectrum`` envelope onto ``wl_um`` (relative units)."""
    wl = np.asarray(wl_um, dtype=float)
    spec = _read_oghma_spectrum(name)
    out = np.asarray(
        [spec.irradiance_at_wavelength_um(float(w)) for w in wl],
        dtype=float,
    )
    return np.maximum(out, 0.0) * float(multiplier)


def _time_support_fs(config: FdtdLightSourceConfig) -> tuple[float, float]:
    if config.t_end_fs > config.t_start_fs and config.t_end_fs > 0:
        pad = max(8.0 * config.tau_fs, 5.0)
        return config.t_start_fs - 0.5 * pad, config.t_end_fs + 0.5 * pad
    pad = max(8.0 * config.tau_fs, 5.0)
    return config.t0_fs - pad, config.t0_fs + pad


def _apply_time_gate(t_fs: np.ndarray, config: FdtdLightSourceConfig) -> np.ndarray:
    gate = np.ones_like(t_fs, dtype=float)
    if config.t_end_fs > config.t_start_fs and config.t_end_fs > 0:
        gate = ((t_fs >= config.t_start_fs) & (t_fs <= config.t_end_fs)).astype(float)
    return gate


def _apply_window(t_fs: np.ndarray, config: FdtdLightSourceConfig) -> np.ndarray:
    wname = config.window.replace("window_", "")
    if wname in ("none", ""):
        return np.ones_like(t_fs, dtype=float)
    if config.t_end_fs <= config.t_start_fs or config.t_end_fs <= 0:
        return np.ones_like(t_fs, dtype=float)
    tau = t_fs - config.t_start_fs
    duration = config.t_end_fs - config.t_start_fs
    if duration <= 0:
        return np.ones_like(t_fs, dtype=float)
    x = np.clip(tau / duration, 0.0, 1.0)
    if wname == "hann":
        w = np.zeros_like(x)
        inside = (tau >= 0) & (tau <= duration)
        w[inside] = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau[inside] / duration))
        return w
    if wname in ("tukey", "window_tukey"):
        alpha = float(config.window_alpha)
        w = np.ones_like(x)
        if alpha <= 0:
            return w
        edge = alpha / 2.0
        left = x < edge
        right = x > (1.0 - edge)
        w[left] = 0.5 * (1.0 + np.cos(np.pi * (x[left] / edge - 1.0)))
        w[right] = 0.5 * (1.0 + np.cos(np.pi * ((x[right] - 1.0 + edge) / edge)))
        return w
    raise NotImplementedError(f"FDTD window {config.window!r} not implemented")


def _waveform_u(t_fs: np.ndarray, config: FdtdLightSourceConfig) -> np.ndarray:
    wf = config.waveform.lower()
    t_s = t_fs * _FS_TO_S
    t0_s = config.t0_fs * _FS_TO_S
    tau_s = max(config.tau_fs * _FS_TO_S, 1e-30)
    dt_s = t_s - t0_s
    omega = 2.0 * math.pi * _C_LIGHT_M_PER_S / (config.carrier_wl_um * 1e-6)

    if wf == "gaus_sin":
        envelope = np.exp(-0.5 * (dt_s / tau_s) ** 2)
        carrier = np.sin(omega * dt_s + config.phase_rad)
        u = envelope * carrier
    elif wf == "cw_sin":
        u = np.sin(omega * t_s + config.phase_rad)
    else:
        raise NotImplementedError(f"FDTD waveform {config.waveform!r} not implemented")

    gate = _apply_time_gate(t_fs, config)
    window = _apply_window(t_fs, config)
    return config.amp * u * gate * window + config.dc_offset * gate


def _fdtd_waveform_fft_on_wl_grid(
    wl_um_grid: np.ndarray,
    config: FdtdLightSourceConfig,
    *,
    dt_fs: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(wl_target, wl_fft_pos, e_f_pos * dt_s)`` for PSD / amplitude helpers."""
    wl_target = np.asarray(wl_um_grid, dtype=float)
    t_min_fs, t_max_fs = _time_support_fs(config)
    n = max(int(math.ceil((t_max_fs - t_min_fs) / dt_fs)) + 1, 4096)
    t_fs = t_min_fs + np.arange(n, dtype=float) * dt_fs
    e_t = _waveform_u(t_fs, config)

    dt_s = dt_fs * _FS_TO_S
    e_f = np.fft.rfft(e_t) * dt_s
    freqs_hz = np.fft.rfftfreq(n, dt_s)

    pos = freqs_hz > 0
    wl_fft = _C_LIGHT_UM_PER_S / freqs_hz[pos]
    e_f_pos = e_f[pos]
    order = np.argsort(wl_fft)
    wl_fft = wl_fft[order]
    e_f_pos = e_f_pos[order]
    return wl_target, wl_fft, e_f_pos


def _interp_complex_on_wl(wl_target: np.ndarray, wl_src: np.ndarray, val_src: np.ndarray) -> np.ndarray:
    real = np.interp(wl_target, wl_src, np.real(val_src), left=0.0, right=0.0)
    imag = np.interp(wl_target, wl_src, np.imag(val_src), left=0.0, right=0.0)
    return np.asarray(real + 1j * imag, dtype=complex)


def compute_fdtd_waveform_fft_amplitude(
    wl_um_grid: np.ndarray,
    config: FdtdLightSourceConfig,
    *,
    dt_fs: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(wl_um, S_FDTD)`` — complex FFT of temporal waveform on ``wl_um_grid``.

    ``S_FDTD`` carries phase; ``|S_FDTD|²`` matches ``compute_fdtd_waveform_psd`` up to interpolation.
    """
    wl_target, wl_fft, e_f_pos = _fdtd_waveform_fft_on_wl_grid(wl_um_grid, config, dt_fs=dt_fs)
    s_fdtd = _interp_complex_on_wl(wl_target, wl_fft, e_f_pos)
    return wl_target, s_fdtd


def compute_fdtd_waveform_psd(
    wl_um_grid: np.ndarray,
    config: FdtdLightSourceConfig,
    *,
    dt_fs: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(wl_um, S_E)`` with ``S_E = |FFT(E(t))|²`` interpolated onto ``wl_um_grid``.

    Uses a dense time grid over the gated pulse support.
    """
    wl_target, wl_fft, e_f_pos = _fdtd_waveform_fft_on_wl_grid(wl_um_grid, config, dt_fs=dt_fs)
    psd_pos = np.abs(e_f_pos) ** 2
    s_e = np.interp(wl_target, wl_fft, psd_pos, left=0.0, right=0.0)
    return wl_target, np.maximum(s_e, 0.0)


def compute_fdtd_source_spectrum_i_new(
    wl_um: np.ndarray,
    config: FdtdLightSourceConfig,
    *,
    dt_fs: float = 0.05,
) -> np.ndarray:
    """``I_new(λ) = S_E(λ) × Intensity(λ)``."""
    wl_arr = np.asarray(wl_um, dtype=float)
    _, s_e = compute_fdtd_waveform_psd(wl_arr, config, dt_fs=dt_fs)
    intensity = load_oghma_light_spectrum(
        config.light_spectrum,
        wl_arr,
        multiplier=config.light_multiplier,
    )
    return s_e * intensity


def predict_fdtd_output_spectrum_incoherent(
    i_new: np.ndarray,
    transfer: np.ndarray,
) -> np.ndarray:
    """Incoherent coupling: ``I_new(λ) × transfer(λ)`` (e.g. ``|t|²``)."""
    return np.asarray(i_new, dtype=float) * np.asarray(transfer, dtype=float)


def compute_fdtd_steady_e_field(
    wl_um: np.ndarray,
    src_config: FdtdLightSourceConfig,
    e_raw: np.ndarray,
    *,
    i_new: np.ndarray | None = None,
    dt_fs: float = 0.05,
    eps: float = 1e-30,
) -> dict[str, np.ndarray]:
    """
    FDTD → steady spatial phasor: ``H = E_raw / S_FDTD``, ``E_steady = H × sqrt(I_new)``.

    ``E_raw`` is the TMM unit-incident response at the target depth (Step 1 analogue).
    """
    wl_arr = np.asarray(wl_um, dtype=float)
    e_raw_arr = np.asarray(e_raw, dtype=complex)
    if e_raw_arr.shape != wl_arr.shape:
        raise ValueError(f"e_raw shape must match wl_um: {e_raw_arr.shape} vs {wl_arr.shape}")

    _, s_fdtd = compute_fdtd_waveform_fft_amplitude(wl_arr, src_config, dt_fs=dt_fs)
    s_fdtd = np.asarray(s_fdtd, dtype=complex)
    denom = np.where(np.abs(s_fdtd) > eps, s_fdtd, eps + 0.0j)
    h_steady = e_raw_arr / denom

    if i_new is None:
        i_new_arr = compute_fdtd_source_spectrum_i_new(wl_arr, src_config, dt_fs=dt_fs)
    else:
        i_new_arr = np.asarray(i_new, dtype=float)
        if i_new_arr.shape != wl_arr.shape:
            raise ValueError(f"i_new shape must match wl_um: {i_new_arr.shape} vs {wl_arr.shape}")

    amp_weight = np.sqrt(np.maximum(i_new_arr, 0.0))
    e_steady = h_steady * amp_weight
    return {
        "e_steady": e_steady,
        "h_steady": h_steady,
        "s_fdtd": s_fdtd,
        "i_new": i_new_arr,
        "amplitude_weight": amp_weight,
    }


def compute_fdtd_steady_e_field_at_z(
    wl_um: np.ndarray,
    src_config: FdtdLightSourceConfig,
    z_um: float,
    build_layers_fn: Callable[[float], list[Any]],
    simulation_module: Any,
    *,
    i_new: np.ndarray | None = None,
    dt_fs: float = 0.05,
    incident_angle_rad: float = 0.0,
    pol: str | None = None,
) -> np.ndarray:
    """Unit TMM ``E_raw(z,λ)`` then :func:`compute_fdtd_steady_e_field` → ``E_steady(z,λ)``."""
    if pol is None:
        pol = "s" if src_config.excite_ey else "p"
    wl_arr = np.atleast_1d(np.asarray(wl_um, dtype=float))
    e_raw_list: list[complex] = []
    for wl in wl_arr:
        layers = build_layers_fn(float(wl))
        e_raw_list.append(
            _cplx_from_py(
                simulation_module.TMM_solver_passive_e_component_at_z_s(
                    layers,
                    float(wl),
                    float(z_um),
                    complex(float(incident_angle_rad), 0.0),
                    "s" if pol == "s" else "p",
                )
            )
        )
    e_raw = np.asarray(e_raw_list, dtype=complex)
    out = compute_fdtd_steady_e_field(
        wl_arr,
        src_config,
        e_raw,
        i_new=i_new,
        dt_fs=dt_fs,
    )
    return out["e_steady"]


def integrate_incoherent_power(
    wl_um: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    """Trapezoidal ``∫ spectrum(λ) dλ`` (wavelength in nm, same units as spectrum density)."""
    wl = np.asarray(wl_um, dtype=float)
    spec = np.asarray(spectrum, dtype=float)
    if wl.size < 2:
        return float(spec[0]) if spec.size else 0.0
    return float(np.trapz(spec, wl))
