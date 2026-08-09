#!/usr/bin/env python3
"""Closed-loop: continuous W^R — gauss_laguerre / residue_series vs Sommerfeld.

Geometry: films_golden (amb|f1|f2|sub), same as sommerfeld_wr_parity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wr_quad_common as wqc


@pytest.fixture(scope="module")
def films_stack():
    return wqc.films_golden_stack()


@pytest.fixture(scope="module")
def W_sommerfeld(films_stack):
    sim = wqc._ensure_sim()
    return wqc.compute_wr(films_stack, quad=sim.wr_quadrature.sommerfeld)


def test_films_golden_gauss_laguerre_vs_sommerfeld(films_stack, W_sommerfeld):
    sim = wqc._ensure_sim()
    W_gl = wqc.compute_wr(
        films_stack,
        quad=sim.wr_quadrature.gauss_laguerre,
        nx_laguerre=wqc.NX_LAGUERRE,
    )
    m = wqc.metrics(W_gl, W_sommerfeld)
    print(
        f"[GL nx={wqc.NX_LAGUERRE}] rel_fro={m['rel_fro']:.6e} "
        f"|ΔW00|={m['abs_w00']:.6e}",
        flush=True,
    )
    wqc.assert_vs_sommerfeld(
        W_gl,
        W_sommerfeld,
        label=f"gauss_laguerre(nx={wqc.NX_LAGUERRE}) vs sommerfeld",
    )


def test_films_golden_residue_series_vs_sommerfeld(films_stack, W_sommerfeld):
    sim = wqc._ensure_sim()
    W_res = wqc.compute_wr(
        films_stack,
        quad=sim.wr_quadrature.residue_series,
        neff_resolution=wqc.RESIDUE_NEFF_RESOLUTION,
    )
    m = wqc.metrics(W_res, W_sommerfeld)
    print(
        f"[residue nres={wqc.RESIDUE_NEFF_RESOLUTION:g}] "
        f"rel_fro={m['rel_fro']:.6e} |ΔW00|={m['abs_w00']:.6e} "
        f"||W||={m['norm_a']:.6e} ||W_som||={m['norm_b']:.6e}",
        flush=True,
    )
    wqc.assert_vs_sommerfeld(
        W_res, W_sommerfeld, label="residue_series vs sommerfeld"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short", "-s"])
