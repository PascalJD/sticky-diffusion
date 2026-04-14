#!/bin/bash

set -u

ENV_PREFIX="${ENV_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sticky}"

if [[ -z "${PYTHON_BIN}" && -n "${ENV_PREFIX}" ]]; then
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
fi

_section() {
  echo
  echo "===== $1 ====="
}

_show_path() {
  local label="$1"
  local path="$2"
  echo "${label}=${path}"
  if [[ -z "${path}" ]]; then
    return
  fi
  if [[ -e "${path}" ]]; then
    ls -ld "${path}" || true
    if command -v readlink >/dev/null 2>&1; then
      echo "resolved_${label}=$(readlink -f "${path}" 2>/dev/null || true)"
    fi
    if command -v file >/dev/null 2>&1; then
      file "${path}" || true
    fi
  else
    echo "${label} is missing"
  fi
}

_run_cmd() {
  local label="$1"
  shift
  _section "${label}"
  echo "+ $*"
  "$@"
  local status=$?
  echo "[exit=${status}]"
  return 0
}

_show_if_exists() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    _section "Contents of ${path}"
    sed -n '1,200p' "${path}" || true
  fi
}

_section "Context"
echo "date=$(date -Is 2>/dev/null || date)"
echo "hostname=$(hostname 2>/dev/null || true)"
echo "pwd=$(pwd)"
echo "user=${USER:-}"
echo "shell=${SHELL:-}"
echo "ENV_PREFIX=${ENV_PREFIX}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "PYTHONHOME=${PYTHONHOME:-}"
echo "PYTHONPATH=${PYTHONPATH:-}"
echo "PATH=${PATH}"

_section "Command discovery"
command -v python || true
command -v python3 || true
command -v conda || true
command -v mamba || true

_section "Primary paths"
_show_path "env_prefix" "${ENV_PREFIX}"
_show_path "python_bin" "${PYTHON_BIN}"

if [[ -n "${ENV_PREFIX}" ]]; then
  _show_path "env_python" "${ENV_PREFIX}/bin/python"
  _show_path "env_activate" "${ENV_PREFIX}/bin/activate"
  _show_path "env_conda_meta" "${ENV_PREFIX}/conda-meta"
  _show_path "env_python_lib" "${ENV_PREFIX}/lib/python3.10"
  _show_path "env_python_zip" "${ENV_PREFIX}/lib/python310.zip"
  _show_path "env_encodings" "${ENV_PREFIX}/lib/python3.10/encodings/__init__.py"
  _show_path "env_site" "${ENV_PREFIX}/lib/python3.10/site.py"
  _show_path "env_os" "${ENV_PREFIX}/lib/python3.10/os.py"
  _show_path "env_dynload" "${ENV_PREFIX}/lib/python3.10/lib-dynload"
fi

_show_if_exists "${ENV_PREFIX}/pyvenv.cfg"
_show_if_exists "${ENV_PREFIX}/conda-meta/history"

if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
  if command -v ldd >/dev/null 2>&1; then
    _run_cmd "ldd on python" ldd "${PYTHON_BIN}"
  fi
  _run_cmd "python version" "${PYTHON_BIN}" -V
  _run_cmd "python startup probe" "${PYTHON_BIN}" -c "import sys; print(sys.executable)"
  _run_cmd "python encodings probe" "${PYTHON_BIN}" -c "import encodings, sys; print(encodings.__file__); print(sys.prefix); print(sys.base_prefix)"
  _run_cmd "python path probe" "${PYTHON_BIN}" -c "import pprint, sys; pprint.pp(sys.path)"
  _run_cmd "python import probe" "${PYTHON_BIN}" -c "import hydra, omegaconf; print('hydra_ok'); print('omegaconf_ok')"
fi

if command -v conda >/dev/null 2>&1; then
  _run_cmd "conda info --envs" conda info --envs
  if [[ -n "${ENV_PREFIX}" && -d "${ENV_PREFIX}" ]]; then
    _run_cmd "conda list -p env_prefix" conda list -p "${ENV_PREFIX}"
  fi
fi
