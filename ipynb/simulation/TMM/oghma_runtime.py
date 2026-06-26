"""Bootstrap ``import simulation`` for Oghma/TMM scripts."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path


def _require_env() -> None:
    for key in (
        "SIMULATION_ARTIFACTS_DIR",
        "SIMULATION_DATABASE_DIR",
        "SIMULATION_TMM_ASSETS_DIR",
    ):
        if not os.environ.get(key, "").strip():
            raise RuntimeError(
                f"{key} unset; source simulation_core/scripts/init-simulation-build-env.sh first"
            )


def tmm_dir() -> Path:
    _require_env()
    path = Path(os.environ["SIMULATION_TMM_ASSETS_DIR"]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"SIMULATION_TMM_ASSETS_DIR 不存在: {path}")
    return path


def simulation_repo_root() -> Path:
    _require_env()
    return Path(os.environ["SIMULATION_DATABASE_DIR"]).resolve().parent.parent


def simulation_runtime_dir() -> Path:
    _require_env()
    return Path(os.environ["SIMULATION_ARTIFACTS_DIR"]).resolve()


def prepare_simulation(*, import_tmm: bool = True) -> Path:
    """Require init env; optionally add TMM to ``sys.path``."""
    _require_env()
    import simulation  # noqa: F401

    runtime = simulation_runtime_dir()
    if import_tmm:
        tmm_s = str(tmm_dir())
        if tmm_s not in sys.path:
            sys.path.insert(0, tmm_s)
    return runtime


def get_simulation_module():
    import simulation

    return simulation


def bootstrap_tmm_session() -> tuple[Path, Path, Path]:
    _require_env()
    import simulation  # noqa: F401
    from simulation_paths import prepare_runtime

    runtime = prepare_runtime(import_simulation=False)
    os.chdir(runtime)
    tmm_s = str(tmm_dir())
    if tmm_s not in sys.path:
        sys.path.insert(0, tmm_s)
    return simulation_repo_root(), runtime, Path(tmm_s)


def init_oghma_test_session() -> tuple[Path, Path]:
    _require_env()
    import simulation  # noqa: F401

    runtime = simulation_runtime_dir()
    os.chdir(runtime)
    return simulation_repo_root(), runtime


@contextmanager
def artifacts_cwd():
    runtime = simulation_runtime_dir()
    orig = os.getcwd()
    try:
        os.chdir(runtime)
        yield runtime
    finally:
        os.chdir(orig)


def oghma_projects_root(repo: Path | None = None) -> Path:
    _require_env()
    import simulation  # noqa: F401
    from simulation_paths import oghma_projects_dir

    return oghma_projects_dir()


def oghma_project_dir(*parts: str, repo: Path | None = None) -> Path:
    return oghma_projects_root(repo).joinpath(*parts)


def tmm_output_dir(subdir: str, repo: Path | None = None) -> Path:
    _require_env()
    import simulation  # noqa: F401
    from simulation_paths import tmm_output_dir as _tmm_output_dir

    return _tmm_output_dir(subdir)


def default_oled_hello_project(repo: Path | None = None) -> Path:
    path = oghma_project_dir("oled", "01_hello_oled", repo=repo)
    if not path.is_dir():
        raise FileNotFoundError(f"OLED project not found: {path}")
    return path
