#!/usr/bin/env python3
"""Tests for Oghma FDTD detector reference CSV loading (oghma_fdtd)."""

from __future__ import annotations

import pytest

from oghma_fdtd import load_oghma_fdtd_detector_reference
from oghma_runtime import init_oghma_test_session

init_oghma_test_session()


def test_load_fdtd_reference_missing_detector_raises(tmp_path):
    """Missing detector CSVs raise FileNotFoundError, not downstream KeyError."""
    with pytest.raises(FileNotFoundError, match="detector0/lam_E.csv"):
        load_oghma_fdtd_detector_reference(tmp_path)
