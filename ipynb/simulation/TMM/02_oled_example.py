#!/usr/bin/env python3
"""OLED JV / EQE / luminance workflow — ours vs Oghma 02_oled_jv baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from oghma_oled_jv_cases import run_oled_jv_plot_workflow
from oghma_oled_utils import DEFAULT_OLED_JV_PROJECT
from oghma_runtime import bootstrap_tmm_session

_, _RUNTIME, _ = bootstrap_tmm_session()
import simulation  # noqa: E402
import simulation_database_parser as sdp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_OLED_JV_PROJECT)
    parser.add_argument(
        "--solver",
        choices=("default", "gummel", "newton"),
        default="default",
        help="OLED DD solver backend for JV optical coupling.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    print(f"RUNTIME={_RUNTIME}")
    print(f"solver={args.solver}")
    print(f"og/materials: {sdp.materials_root(init=True)}")
    print(f"project: {args.project}")

    out_dir = run_oled_jv_plot_workflow(
        project_path=args.project,
        solver=args.solver,
        out_dir=args.out_dir,
        simulation_module=simulation,
    )
    print(f"Wrote plots to {out_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
