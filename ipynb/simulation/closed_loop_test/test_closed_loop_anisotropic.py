#!/usr/bin/env python3
"""Closed-loop case 3: anisotropic film + soft RCWA (Floquet) + diffraction smoke."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import closed_loop_common as clc
import simulation as sim

# In-plane birefringence + off-diag: cross-pol r,t and flux≠|r|² visible on plots.
# Lab abcde=(ε_xx,ε_xy,ε_yx,ε_yy,ε_zz); xy-block PD: |s|<√(ε_xx ε_yy)≈√6≈2.45.
_ANISO_ABCDE = (2.0 + 0j, 0.35 + 0j, 0.35 + 0j, 3.0 + 0j, 2.25 + 0j)
_ANISO_DEPTH_UM = 0.25


def _aniso_stack(d: float = _ANISO_DEPTH_UM):
    air = clc.make_iso_layer(1.0 + 0j, 0.0, "air")
    mat = sim.material_s.from_anisotropic_nk(list(_ANISO_ABCDE), "aniso_xy")
    film = sim.layer_s()
    film.background_material = sim.share_material_s(mat)
    film.depth = float(d)
    sub = clc.make_iso_layer(1.5 + 0j, 0.0, "sub")
    return [air, film, sub]


def _aniso_stack_d(d: float = _ANISO_DEPTH_UM):
    """Same lab abcde as _aniso_stack, as layer_d for RCWA (ε₂ embed is inside RCWA)."""
    top = sim.make_layer_from_nk_d(1.0 + 0j, 0.0, "air")
    mat = sim.material_d.from_anisotropic_nk(list(_ANISO_ABCDE), "aniso_xy")
    film = sim.layer_d()
    film.background_material = sim.share_material_d(mat)
    film.depth = float(d)
    bot = sim.make_layer_from_nk_d(1.5 + 0j, 0.0, "sub")
    return [top, film, bot]


def _soft_shape_layers_d(shape_kind: str, bg_nk: complex = 1.5 + 0j):
    """Soft structure: structure material == background."""
    bg = sim.share_material_d(sim.material_d.from_nk(bg_nk, "bg"))
    st = sim.share_material_d(sim.material_d.from_nk(bg_nk, "st"))
    top = sim.make_layer_from_nk_d(1.0 + 0j, 0.0, "air")
    film = sim.layer_d()
    film.background_material = bg
    film.depth = 0.25
    film.parent = [-1]
    if shape_kind == "circle":
        film.shapes = [sim.make_circle_shape_d(0.2, [0.0, 0.0], 0.0, st)]
    elif shape_kind == "ellipse":
        film.shapes = [sim.make_ellipse_shape_d([0.26, 0.14], [0.0, 0.0], 0.0, st)]
    elif shape_kind == "rect":
        film.shapes = [sim.make_rect_shape_d([0.2, 0.12], [0.0, 0.0], 0.0, st)]
    elif shape_kind == "poly":
        verts = [[-0.22, -0.16], [0.22, -0.16], [0.22, 0.00], [0.00, 0.00], [0.00, 0.26], [-0.22, 0.26]]
        film.shapes = [sim.make_poly_shape_d(verts, [0.0, 0.0], 0.0, st)]
    elif shape_kind == "bezier_poly_2":
        p0, p1, p2 = [-0.16, -0.12], [0.18, -0.12], [0.02, 0.22]
        cps = [[p0, [0.01, -0.22], p1], [p1, [0.28, 0.08], p2], [p2, [-0.20, 0.10], p0]]
        film.shapes = [sim.make_bezier_poly_2_shape_d(cps, [0.0, 0.0], 0.0, st)]
    elif shape_kind == "concentric_circles":
        film.shapes = [
            sim.make_concentric_circles_shape_d([0.07, 0.16, 0.26], [st, bg, st], [0.0, 0.0], 0.0)
        ]
    else:
        pytest.skip(f"shape {shape_kind} not wired")
    bot = sim.make_layer_from_nk_d(1.5 + 0j, 0.0, "sub")
    return [top, film, bot]


def _uniform_layers_d(film_nk: complex = 1.5 + 0j, depth: float = 0.25):
    return [
        sim.make_layer_from_nk_d(1.0 + 0j, 0.0, "air"),
        sim.make_layer_from_nk_d(film_nk, depth, "film"),
        sim.make_layer_from_nk_d(1.5 + 0j, 0.0, "sub"),
    ]


def _harmonic_G_1d(N: int):
    return [[m, 0] for m in range(-int(N), int(N) + 1)]


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_aniso_vs_rcwa(pol: str):
    """Aniso film: co+cross r,t; R,T = total Sz flux; |E| with chef-ray lab Jones."""
    clc.require_rcwa_triad()
    angles = clc.angle_grid_rad(17)
    aniso = clc.sweep_smm_aniso(
        _aniso_stack(),
        pol=pol,
        angles_rad=angles,
        layer_index=1,
        depth_in_layer=0.0,
        with_cross=True,
        name="SMM_aniso",
    )
    rcwa = clc.sweep_rcwa_per_angle(
        _aniso_stack_d(),
        pol=pol,
        angles_rad=angles,
        which_layer=1,
        z_offset=0.0,
        with_cross=True,
        name="RCWA",
    )
    assert np.all(np.isfinite(aniso.R)) and np.all(np.isfinite(aniso.T))
    assert float(np.max(aniso.R)) <= 1.05 + 1e-6
    clc.assert_sweep_vs_ref(
        aniso, rcwa, rt_tol=clc.RCWA_RT_TOL, e_tol=clc.E_TOL, cross_tol=clc.RCWA_RT_TOL
    )
    clc.plot_solver_suite([aniso, rcwa], pol=pol, title_prefix="aniso_film")


@pytest.mark.parametrize("pol", ["s", "p"])
def test_closed_loop_rcwa_soft_vs_uniform_overlay(pol: str):
    """Soft shapes (structure==bg) must match uniform RCWA; plot both + Δ."""
    clc.require_rcwa_triad()
    angles = clc.angle_grid_rad(17)
    soft = clc.sweep_rcwa_per_angle(
        _soft_shape_layers_d("circle"),
        pol=pol,
        angles_rad=angles,
        which_layer=1,
        z_offset=0.0,
        name="RCWA_soft_circle",
    )
    uniform = clc.sweep_rcwa_per_angle(
        _uniform_layers_d(1.5 + 0j, 0.25),
        pol=pol,
        angles_rad=angles,
        which_layer=1,
        z_offset=0.0,
        name="RCWA_uniform",
    )
    # Soft≡bg: RT only (field grid may differ by FFT/shape sampling).
    clc.assert_close_arr(uniform.R, soft.R, clc.RT_TOL, "soft.R")
    clc.assert_close_arr(uniform.T, soft.T, clc.RT_TOL, "soft.T")
    clc.assert_close_arr(uniform.r, soft.r, clc.RT_TOL, "soft.r")
    clc.assert_close_arr(uniform.t, soft.t, clc.RT_TOL, "soft.t")
    clc.plot_solver_suite([uniform, soft], pol=pol, title_prefix="rcwa_soft_vs_uniform")


@pytest.mark.parametrize(
    "shape",
    ["circle", "ellipse", "rect", "poly", "bezier_poly_2", "concentric_circles"],
)
def test_closed_loop_rcwa_soft_shape_per_angle(shape: str):
    """Soft shapes with n_G=1: per-angle rebuild; R+T finite and near conservation."""
    clc.require_rcwa_triad()
    layers = _soft_shape_layers_d(shape)
    for th_deg in (0.0, 15.0, 30.0):
        R, T = sim.RCWA_get_r_t_power(layers, clc.WL_UM, n_G=1, theta_deg=th_deg, pol="s")
        R, T = float(R), float(T)
        assert math.isfinite(R) and math.isfinite(T)
        # Soft structure==bg → lossless conservation at machine precision.
        assert abs(R + T - 1.0) < clc.RT_TOL


def test_closed_loop_rcwa_floquet_angle_sweep():
    """Floquet single-eigensolve strategy (case 3 angle path): drive-order sweep + plot."""
    if not hasattr(sim, "make_rcwa_floquet_session_d"):
        pytest.skip("Floquet session API missing")
    layers = _uniform_layers_d(1.5 + 0j, 0.25)
    wl = 0.55
    Lr = [1.0, 0.0, 0.0, 1.0]
    N = 2
    G = _harmonic_G_1d(N)
    k_inc = [0.0, 0.0]
    sess = sim.make_rcwa_floquet_session_d(layers, wl, Lr, G, k_inc)
    assert int(sess.n()) == 2 * N + 1
    # FFT G-window is (nx/2 - nx, nx/2]: need nx > 2N so ±N both land (even nx drops -nx/2).
    nxy = [2 * N + 1, 2 * N + 1]
    ms = list(range(-N, N + 1))
    Eamp = []
    Ephase = []
    for m in ms:
        E, _H = sess.fields_on_grid(nxy, [m, 0], pol="s", which_layer=1, z_offset=0.0)
        arr = np.asarray(E)
        assert arr.shape[-1] == 3
        assert np.all(np.isfinite(arr))
        ea, ep = clc.e_amp_phase(arr[0, 0, 0], arr[0, 0, 1], arr[0, 0, 2])
        Eamp.append(ea)
        Ephase.append(ep)
    # Propagating ±1 must both appear once the grid admits negative Nyquist.
    assert Eamp[ms.index(-1)] > 0.1 and Eamp[ms.index(1)] > 0.1
    assert abs(Eamp[ms.index(-1)] - Eamp[ms.index(1)]) < 1e-6
    # Reuse SweepResult: x-axis = drive order m (stored in angles_deg slot).
    floq = clc.SweepResult(
        "RCWA_floquet",
        np.asarray(ms, dtype=float),
        np.zeros(len(ms), dtype=np.complex128),
        np.zeros(len(ms), dtype=np.complex128),
        np.zeros(len(ms), dtype=float),
        np.zeros(len(ms), dtype=float),
        np.asarray(Eamp, dtype=float),
        np.asarray(Ephase, dtype=float),
    )
    try:
        from py_core_plugins.plot_curves import plot_curves
    except ImportError:
        from plot_curves import plot_curves
    import matplotlib.pyplot as plt

    plot_curves(
        [floq.Eamp.tolist(), floq.Ephase.tolist()],
        x_data=[floq.angles_deg.tolist(), floq.angles_deg.tolist()],
        legends=["Eamp", "Ephase"],
        types=["-o", "--s"],
        sample_rate=1.0,
        title="rcwa_floquet_drive_order s-field",
        xlabel="drive order m",
        ylabel="E",
    )
    plt.close("all")


def test_closed_loop_rcwa_diffraction_smoke():
    """Hard structure material ≠ background: per-order flux finite."""
    if not hasattr(sim, "rcwa_poynting_by_order_d"):
        pytest.skip("rcwa_poynting_by_order_d missing")
    bg = sim.share_material_d(sim.material_d.from_nk(1.5 + 0j, "bg"))
    st = sim.share_material_d(sim.material_d.from_nk(2.0 + 0j, "st"))
    layers = [
        sim.make_layer_from_nk_d(1.0 + 0j, 0.0, "air"),
        sim.layer_d(),
        sim.make_layer_from_nk_d(1.5 + 0j, 0.0, "sub"),
    ]
    layers[1].background_material = bg
    layers[1].depth = 0.25
    layers[1].parent = [-1]
    layers[1].shapes = [sim.make_circle_shape_d(0.2, [0.0, 0.0], 0.0, st)]
    fwd, back, _G = sim.rcwa_poynting_by_order_d(
        layers, 0.55, n_G=1, theta_deg=10.0, pol="s", which_layer=0
    )
    assert len(fwd) >= 1
    assert all(math.isfinite(abs(complex(x))) for x in list(fwd) + list(back))


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
