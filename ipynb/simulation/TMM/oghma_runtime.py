"""Bootstrap ``import simulation`` for Oghma/TMM scripts."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

_TMM_RUN_HINT = (
    "source simulation_core/scripts/init-simulation-build-env.sh [build|build-asan], "
    "then assets/ipynb/simulation/TMM/run_tmm.sh test|jupyter"
)


def _require_runtime_env() -> None:
    for key in (
        "SIMULATION_ARTIFACTS_DIR",
        "SIMULATION_DATABASE_DIR",
    ):
        if not os.environ.get(key, "").strip():
            raise RuntimeError(f"{key} unset; {_TMM_RUN_HINT}")


def tmm_dir() -> Path:
    return Path(__file__).resolve().parent


def simulation_runtime_dir() -> Path:
    from simulation_paths import resolve_artifacts_dir

    return resolve_artifacts_dir()


def bootstrap_tmm_session(*, import_tmm: bool = True) -> tuple[Path, Path] | tuple[Path, Path, Path]:
    """Require init env; import simulation; chdir to artifacts; optionally add TMM to ``sys.path``.

    Returns ``(repo_root, runtime_artifacts_dir)`` or ``(repo_root, runtime_artifacts_dir, tmm_dir)``.
    """
    _require_runtime_env()
    import simulation  # noqa: F401
    from simulation_paths import resolve_artifacts_dir

    runtime = resolve_artifacts_dir()
    os.chdir(runtime)
    repo = Path(os.environ["SIMULATION_DATABASE_DIR"]).resolve().parent.parent
    if not import_tmm:
        return repo, runtime
    tmm_s = str(tmm_dir())
    if tmm_s not in sys.path:
        sys.path.insert(0, tmm_s)
    return repo, runtime, Path(tmm_s)


@contextmanager
def artifacts_cwd():
    runtime = simulation_runtime_dir()
    orig = os.getcwd()
    try:
        os.chdir(runtime)
        yield runtime
    finally:
        os.chdir(orig)


def oghma_projects_root() -> Path:
    env_root = os.environ.get("OGHMA_PROJECTS_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    db = os.environ.get("SIMULATION_DATABASE_DIR", "").strip()
    if db:
        local = Path(db) / "og" / "oghma_projects"
        if local.is_dir():
            return local.resolve()
    return (tmm_dir() / ".." / ".." / ".." / "oghma_projects").resolve()


def oghma_project_dir(*parts: str) -> Path:
    return oghma_projects_root().joinpath(*parts)


def tmm_output_dir(subdir: str) -> Path:
    return tmm_dir() / "output" / subdir


DEFAULT_OLED_HELLO_PROJECT = oghma_project_dir("oled", "01_hello_oled")


def default_oled_hello_project() -> Path:
    path = DEFAULT_OLED_HELLO_PROJECT
    if not path.is_dir():
        raise FileNotFoundError(f"OLED project not found: {path}")
    return path
