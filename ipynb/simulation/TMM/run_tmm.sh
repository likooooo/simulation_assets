#!/usr/bin/env bash
# TMM / Oghma alignment: require simulation_core runtime env, then test or jupyter.
#
# Prerequisite (manual, every shell):
#   source simulation_core/scripts/init-simulation-build-env.sh [build|build-asan]
#   pip install -r simulation_core/3rdparty/infrastructure/requirements.txt
#
# TMM 属于 simulation_core 工作流，不依赖 simulation_toykits / init-toykits-build-env.sh。
#
# Usage:
#   ./run_tmm.sh [test] [pytest args...]   # pip install -r requirements.txt; pytest; then other *.py
#   ./run_tmm.sh jupyter [jupyter args...]
#
# Oghma 项目数据：先运行 database/update_all.py --only oghma_projects
# （scp OneDrive → database/og/oghma_projects），再跑 TMM。

set -euo pipefail

_TMM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_INIT_HINT="source simulation_core/scripts/init-simulation-build-env.sh [build|build-asan]"

require_runtime_env() {
  if [[ -z "${SIMULATION_ARTIFACTS_DIR:-}" || -z "${SIMULATION_DATABASE_DIR:-}" ]]; then
    echo "run_tmm.sh: runtime env not set; ${_INIT_HINT}" >&2
    exit 1
  fi
}

require_simulation_core_database() {
  local db="${SIMULATION_DATABASE_DIR}"
  if [[ ! -d "${db}/og/materials" ]]; then
    echo "run_tmm.sh: SIMULATION_DATABASE_DIR must be simulation_core YAML 材料库 (含 og/materials/)" >&2
    echo "  当前: ${db}" >&2
    echo "  请勿使用 init-toykits-build-env.sh；请改用: ${_INIT_HINT}" >&2
    exit 1
  fi
  if [[ ! -f "${SIMULATION_ARTIFACTS_DIR}/simulation.so" ]]; then
    echo "run_tmm.sh: SIMULATION_ARTIFACTS_DIR 须指向 simulation_core build 目录 (含 simulation.so)" >&2
    echo "  当前: ${SIMULATION_ARTIFACTS_DIR}" >&2
    echo "  请改用: ${_INIT_HINT}" >&2
    exit 1
  fi
}

resolve_oghma_projects_root() {
  if [[ -n "${OGHMA_PROJECTS_ROOT:-}" ]]; then
    return 0
  fi
  local local_root="${SIMULATION_DATABASE_DIR}/og/oghma_projects"
  if [[ -d "${local_root}" ]]; then
    export OGHMA_PROJECTS_ROOT="${local_root}"
    echo ">>> run_tmm.sh: OGHMA_PROJECTS_ROOT=${OGHMA_PROJECTS_ROOT}" >&2
    return 0
  fi
  echo "run_tmm.sh: ${local_root} not found; run: python database/update_all.py --only oghma_projects" >&2
  return 1
}

prepare_tmm_env() {
  require_runtime_env
  require_simulation_core_database
  resolve_oghma_projects_root
  cd "${_TMM_DIR}"
  install_tmm_requirements
}

install_tmm_requirements() {
  local req="${_TMM_DIR}/requirements.txt"
  if [[ ! -f "${req}" ]]; then
    echo "run_tmm.sh: missing ${req}" >&2
    exit 1
  fi
  echo ">>> run_tmm.sh: python3 -m pip install -r requirements.txt" >&2
  python3 -m pip install -r "${req}"
}

run_tmm_tests() {
  python3 -m pytest test_*.py "$@"
}

run_tmm_scripts() {
  local script failed=0
  shopt -s nullglob
  for script in *.py; do
    [[ "${script}" == test_* ]] && continue
    echo ">>> run_tmm.sh: python3 ${script}" >&2
    if ! python3 "${script}"; then
      echo "run_tmm.sh: ${script} failed" >&2
      failed=1
    fi
  done
  shopt -u nullglob
  return "${failed}"
}

run_tmm_test_suite() {
  run_tmm_tests "$@" || return 1
  run_tmm_scripts || return 1
}

case "${1:-test}" in
  test)
    shift || true
    prepare_tmm_env
    run_tmm_test_suite "$@"
    ;;
  jupyter)
    shift
    prepare_tmm_env
    exec jupyter notebook "$@"
    ;;
  -h|--help)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    prepare_tmm_env
    run_tmm_test_suite "$@"
    ;;
esac
