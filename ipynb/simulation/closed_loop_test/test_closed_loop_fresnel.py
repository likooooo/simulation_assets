#!/usr/bin/env python3
"""Closed-loop case 1: half-space Fresnel — TMM / SMM / SMM_aniso."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import closed_loop_common as clc


def _glass_air():
    return [clc.make_iso_layer(1.5 + 0j, 0.0, "glass"), clc.make_iso_layer(1.0 + 0j, 0.0, "air")]


def _glass_air_aniso():
    return [clc.make_aniso_diag_layer(1.5 + 0j, 0.0, "glass"), clc.make_aniso_diag_layer(1.0 + 0j, 0.0, "air")]


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_fresnel_glass_to_air(pol: str):
    angles = clc.angle_grid_rad()
    layers = _glass_air()
    tmm, smm = clc.sweep_tmm_smm(layers, pol=pol, angles_rad=angles)
    aniso = clc.sweep_smm_aniso(_glass_air_aniso(), pol=pol, angles_rad=angles, name="SMM_aniso")
    clc.assert_sweep_vs_ref(tmm, smm)
    # >1e-6: diag-ε aniso vs TMM peak ~2e-6 (float / q-branch).
    clc.assert_sweep_vs_ref(tmm, aniso, rt_tol=clc.ANISO_TOL, e_tol=clc.ANISO_TOL)
    clc.plot_solver_suite([tmm, smm, aniso], pol=pol, title_prefix="fresnel glass→air")


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_fresnel_air_to_glass(pol: str):
    angles = clc.angle_grid_rad()
    layers = [clc.make_iso_layer(1.0 + 0j, 0.0, "air"), clc.make_iso_layer(1.5 + 0j, 0.0, "glass")]
    aniso_layers = [
        clc.make_aniso_diag_layer(1.0 + 0j, 0.0, "air"),
        clc.make_aniso_diag_layer(1.5 + 0j, 0.0, "glass"),
    ]
    tmm, smm = clc.sweep_tmm_smm(layers, pol=pol, angles_rad=angles)
    aniso = clc.sweep_smm_aniso(aniso_layers, pol=pol, angles_rad=angles, name="SMM_aniso")
    clc.assert_sweep_vs_ref(tmm, smm)
    clc.assert_sweep_vs_ref(tmm, aniso, rt_tol=clc.ANISO_TOL, e_tol=clc.ANISO_TOL)
    clc.plot_solver_suite([tmm, smm, aniso], pol=pol, title_prefix="fresnel air→glass")


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
