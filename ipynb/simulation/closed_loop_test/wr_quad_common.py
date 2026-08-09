#!/usr/bin/env python3
"""Helpers for continuous W^R quadrature closed-loop (Sommerfeld baseline)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# films_golden geometry (matches sommerfeld_wr_parity / former test_sommerfeld_wr_vs_smuthi)
WL = 500.0
Z_ABOVE = 200.0
L_MAX = 2
M_MAX = 2
N_MED = 1.0
NEFF_IMAG = 1e-2
NEFF_MAX = 0.0
# Sommerfeld baseline matches sommerfeld_wr_parity / former C++ golden.
NEFF_RESOLUTION = 2e-3
# Residue pole-aware Γ needs denser sampling to match the Sommerfeld integral
# at films_golden (2e-3 → ~1.2e-2 rel; 1e-3 → ~3.8e-4).
RESIDUE_NEFF_RESOLUTION = 1e-3
NX_LAGUERRE = 48
WR_POLE_IM_MAX = 0.2
# Explicit (not -1): Sommerfeld/residue method defaults differ under -1.
AUTO_NEFF_CAP_FROM_Z = 0

REL_FRO_TOL = 1e-3
ABS_W00_TOL = 1e-4


def _ensure_sim() -> Any:
    art = os.environ.get("SIMULATION_ARTIFACTS_DIR", "").strip()
    if not art:
        default = os.path.expanduser(
            "~/repos/simulation_toykits/simulation_core/build"
        )
        if os.path.isfile(os.path.join(default, "simulation.so")):
            os.environ["SIMULATION_ARTIFACTS_DIR"] = default
            art = default
    if art:
        import sys

        if art not in sys.path:
            sys.path.insert(0, art)
    import simulation as sim

    return sim


def films_golden_stack() -> list[Any]:
    """Top→bottom: amb(1,0) | f1(1.6,40) | f2(2.0,30) | sub(1.5,0)."""
    sim = _ensure_sim()
    return [
        sim.make_layer_from_nk_d(complex(N_MED), 0.0, "amb"),
        sim.make_layer_from_nk_d(1.6 + 0j, 40.0, "f1"),
        sim.make_layer_from_nk_d(2.0 + 0j, 30.0, "f2"),
        sim.make_layer_from_nk_d(1.5 + 0j, 0.0, "sub"),
    ]


def compute_wr(
    stack: list[Any] | None = None,
    *,
    quad: Any | None = None,
    wl: float = WL,
    z: float = Z_ABOVE,
    l_max: int = L_MAX,
    m_max: int = M_MAX,
    n_med: float = N_MED,
    neff_imag: float = NEFF_IMAG,
    neff_max: float = NEFF_MAX,
    neff_resolution: float = NEFF_RESOLUTION,
    nx_laguerre: int = NX_LAGUERRE,
    wr_pole_im_max: float = WR_POLE_IM_MAX,
    auto_neff_cap_from_z: int = AUTO_NEFF_CAP_FROM_Z,
) -> np.ndarray:
    sim = _ensure_sim()
    if stack is None:
        stack = films_golden_stack()
    if quad is None:
        quad = sim.wr_quadrature.sommerfeld
    omega = 2.0 * np.pi / float(wl)
    k_med = complex(n_med) * omega
    W = sim.compute_layer_mediated_coupling_self_d(
        stack,
        k_med,
        float(wl),
        float(z),
        float(z),
        int(l_max),
        int(m_max),
        quad,
        float(neff_imag),
        float(neff_max),
        float(neff_resolution),
        int(nx_laguerre),
        float(wr_pole_im_max),
        int(auto_neff_cap_from_z),
    )
    return np.asarray(W, dtype=np.complex128)


def metrics(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-30) -> dict[str, float]:
    """Return rel_fro and |ΔW00| of a vs baseline b."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    diff = a - b
    denom = max(float(np.linalg.norm(b)), eps)
    return {
        "rel_fro": float(np.linalg.norm(diff) / denom),
        "abs_w00": float(abs(complex(diff[0, 0]))),
        "norm_a": float(np.linalg.norm(a)),
        "norm_b": float(np.linalg.norm(b)),
    }


def assert_vs_sommerfeld(
    W: np.ndarray,
    W_som: np.ndarray,
    *,
    rel_tol: float = REL_FRO_TOL,
    abs00_tol: float = ABS_W00_TOL,
    label: str = "",
) -> dict[str, float]:
    m = metrics(W, W_som)
    tag = label or "vs sommerfeld"
    assert m["rel_fro"] <= rel_tol, (
        f"{tag}: rel_fro={m['rel_fro']:.6e} > {rel_tol:.3e} "
        f"(||W||={m['norm_a']:.6e}, ||W_som||={m['norm_b']:.6e})"
    )
    assert m["abs_w00"] <= abs00_tol, (
        f"{tag}: |ΔW00|={m['abs_w00']:.6e} > {abs00_tol:.3e}"
    )
    return m
