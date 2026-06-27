#!/usr/bin/env python3
"""Regression tests for unified TMM z / gpvdm-Oghma y coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oghma_runtime import get_simulation_module, init_oghma_test_session, oghma_project_dir

init_oghma_test_session()
simulation = get_simulation_module()

from oghma_core import (
    _structure_pos_from_layers,
    gpvdm_y_to_z_simulation_um,
    load_oghma_project,
    tmm_stack_z_edges_from_bar_thicknesses,
    z_simulation_to_gpvdm_y_um,
)
from oghma_oled_utils import build_oghma_oled_passive_stack_ito_al

_OLED_PROJECT = oghma_project_dir("oled", "01_hello_oled")


def _require_oled_project() -> Path:
    if not _OLED_PROJECT.is_dir():
        pytest.skip(f"OLED project not found: {_OLED_PROJECT}")
    return _OLED_PROJECT


def test_gpvdm_y_z_identity() -> None:
    y = np.linspace(0.0, 0.5, 5)
    np.testing.assert_allclose(gpvdm_y_to_z_simulation_um(y), y)
    np.testing.assert_allclose(z_simulation_to_gpvdm_y_um(y), y)


def test_tmm_stack_z_edges_from_bar_thicknesses() -> None:
    edges, _ = tmm_stack_z_edges_from_bar_thicknesses([0.1, 0.2, 0.05])
    assert edges[0] == 0.0
    np.testing.assert_allclose(edges[-1], 0.35)


def test_oled_project_structure_pos() -> None:
    project_dir = _require_oled_project()
    project = load_oghma_project(project_dir)
    layers = build_oghma_oled_passive_stack_ito_al(project, 0.55, simulation)
    pos = _structure_pos_from_layers(layers)
    assert len(pos) == len(layers) + 1
    assert pos[1] == 0.0
