#!/usr/bin/env python3
"""Interface tests for OLED JV / EQE / luminance (02_oled_jv baseline)."""

from __future__ import annotations

import sys

from oghma_oled_jv_cases import main_jv_cli


if __name__ == "__main__":
    sys.exit(main_jv_cli())
