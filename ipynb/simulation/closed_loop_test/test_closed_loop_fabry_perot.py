#!/usr/bin/env python3
"""Closed-loop case 2: Fabry–Pérot — TMM / SMM / SMM_aniso / RCWA (n_G=1, per-angle)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import closed_loop_common as clc  # loads db/mp before simulation
import simulation as sim

try:
    from simulation_database_parser import materials_db_from_token_paths
    from coating_visualizer import _make_coating_from_material, layers_from_formula
except ImportError:
    materials_db_from_token_paths = None  # type: ignore
    layers_from_formula = None  # type: ignore

# 与 simulation_toykits/filmstack_templates.json id=fabry_perot 对齐（测试自包含，不依赖父仓 JSON）
_FABRY_PEROT_MATERIAL_PATH_KEYS = [
    ["rii", "materials", "other", "mixed_gases", "air_Ciddor.yml"],
    ["rii", "materials", "main", "Si", "Si_Aspnes.yml"],
]
_FABRY_PEROT_FORMULA = (
    "air_Ciddor 0 Mirror 0.01 3.0 0 air_Ciddor 0.25 Mirror 0.01 3.0 0 Si_Aspnes 0"
)


def _fabry_perot_layers_s():
    if materials_db_from_token_paths is None or layers_from_formula is None:
        pytest.skip("coating_visualizer / simulation_database_parser not importable")
    path_entries = {}
    for keys in _FABRY_PEROT_MATERIAL_PATH_KEYS:
        token = Path(keys[-1]).stem
        path_entries[token] = keys
    mdb = materials_db_from_token_paths(path_entries)
    mats, depths = layers_from_formula(
        _FABRY_PEROT_FORMULA, mdb, simulation_module=sim
    )
    layers = []
    for m, d in zip(mats, depths):
        layers.append(_make_coating_from_material(sim, m, float(d)))
    return layers


def _aniso_diag_copy(layers_s):
    from coating_solver_test_util import coating_nk_at_wl

    out = []
    for lyr in layers_s:
        n = coating_nk_at_wl(lyr.background_material, clc.WL_UM)
        out.append(clc.make_aniso_diag_layer(n, float(lyr.depth), lyr.background_material.name()))
    return out


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_fabry_perot(pol: str):
    clc.require_rcwa_triad()
    layers = _fabry_perot_layers_s()
    angles = clc.angle_grid_rad(21)
    tmm, smm = clc.sweep_tmm_smm(layers, pol=pol, angles_rad=angles, layer_index=1, depth_in_layer=0.0)
    aniso = clc.sweep_smm_aniso(
        _aniso_diag_copy(layers),
        pol=pol,
        angles_rad=angles,
        layer_index=1,
        depth_in_layer=0.0,
        name="SMM_aniso",
    )
    rcwa = clc.sweep_rcwa_per_angle(
        clc.layers_to_d_nk(layers), pol=pol, angles_rad=angles, which_layer=1, z_offset=0.0, name="RCWA"
    )
    clc.assert_sweep_vs_ref(tmm, smm)
    # >1e-6: ANISO_TOL (~2e-6 peak float / q-branch)
    clc.assert_sweep_vs_ref(tmm, aniso, rt_tol=clc.ANISO_TOL, e_tol=clc.ANISO_TOL)
    # >1e-6: RCWA_RT_TOL (~1e-6 peak on r)
    clc.assert_sweep_vs_ref(tmm, rcwa, rt_tol=clc.RCWA_RT_TOL)
    clc.plot_solver_suite([tmm, smm, aniso, rcwa], pol=pol, title_prefix="fabry_perot")


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
