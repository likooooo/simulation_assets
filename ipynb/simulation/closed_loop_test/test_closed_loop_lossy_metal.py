#!/usr/bin/env python3
"""Closed-loop case 4: lossy / metal multilayer — TMM / SMM / SMM_aniso / RCWA (n_G=1, per-angle)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import closed_loop_common as clc


# 源自 filmstack_templates.json id=paper_sio2_si 的 nk；半无限有损基底在功率账本中
# 仍可 R+T≈1，故改为有限有损 Si 膜 + 无损衬底，使吸收出现在 R+T 缺口中。
def _lossy_sio2_si():
    return [
        clc.make_iso_layer(1.0 + 0j, 0.0, "air"),
        clc.make_iso_layer(1.46 + 0j, 0.05, "SiO2"),
        clc.make_iso_layer(3.87 + 0.02j, 0.2, "Si_lossy"),
        clc.make_iso_layer(1.5 + 0j, 0.0, "glass"),
    ]


# 与 simulation_toykits/filmstack_templates.json id=spr_bk7_cr_au 对齐（测试自包含，不依赖父仓 JSON）
def _metal_spr():
    return [
        clc.make_iso_layer(1.517 + 0j, 0.0, "BK7"),
        clc.make_iso_layer(3.719 + 4.362j, 0.005, "Cr"),
        clc.make_iso_layer(0.13 + 3.162j, 0.03, "Au"),
        clc.make_iso_layer(1.0 + 0j, 0.0, "air"),
    ]


def _aniso_diag_copy(layers_s):
    from coating_solver_test_util import coating_nk_at_wl

    out = []
    for lyr in layers_s:
        n = coating_nk_at_wl(lyr.background_material, clc.WL_UM)
        out.append(clc.make_aniso_diag_layer(n, float(lyr.depth), lyr.background_material.name()))
    return out


def _run_closed_loop_suite(
    layers,
    pol: str,
    title_prefix: str,
    *,
    layer_index: int = 1,
    rcwa_rt_tol: float = clc.RCWA_RT_TOL,
):
    clc.require_rcwa_triad()
    angles = clc.angle_grid_rad(21)
    tmm, smm = clc.sweep_tmm_smm(
        layers, pol=pol, angles_rad=angles, layer_index=layer_index, depth_in_layer=0.0
    )
    aniso = clc.sweep_smm_aniso(
        _aniso_diag_copy(layers),
        pol=pol,
        angles_rad=angles,
        layer_index=layer_index,
        depth_in_layer=0.0,
        name="SMM_aniso",
    )
    rcwa = clc.sweep_rcwa_per_angle(
        clc.layers_to_d_nk(layers),
        pol=pol,
        angles_rad=angles,
        which_layer=layer_index,
        z_offset=0.0,
        name="RCWA",
    )
    # Absorption channel open: R+T must drop below lossless conservation.
    rt_sum = np.real(np.asarray(tmm.R, dtype=np.complex128)) + np.real(
        np.asarray(tmm.T, dtype=np.complex128)
    )
    assert float(np.max(rt_sum)) < 1.0 - 1e-3, f"{title_prefix}: expected absorption (max R+T={float(np.max(rt_sum))})"
    clc.assert_sweep_vs_ref(tmm, smm)
    # >1e-6: ANISO_TOL (~2e-6 peak float / q-branch)
    clc.assert_sweep_vs_ref(tmm, aniso, rt_tol=clc.ANISO_TOL, e_tol=clc.ANISO_TOL)
    clc.assert_sweep_vs_ref(tmm, rcwa, rt_tol=rcwa_rt_tol)
    clc.plot_solver_suite([tmm, smm, aniso, rcwa], pol=pol, title_prefix=title_prefix)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_lossy_sio2_si(pol: str):
    _run_closed_loop_suite(_lossy_sio2_si(), pol, "lossy_sio2_si", layer_index=1)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_metal_spr(pol: str):
    # >1e-6: SPR metal p-pol RCWA peak |Δt|~1.5e-5 (n_G=1 uniform)
    _run_closed_loop_suite(
        _metal_spr(), pol, "metal_spr", layer_index=1, rcwa_rt_tol=2e-5
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
