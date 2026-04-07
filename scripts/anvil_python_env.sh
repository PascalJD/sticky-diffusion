#!/bin/bash

sticky_resolve_python_env() {
  local env_prefix="${ENV_PREFIX:-}"
  local python_bin="${PYTHON_BIN:-}"
  local conda_env_name="${CONDA_ENV_NAME:-sticky}"

  # A stale PYTHONHOME can break interpreter startup before our code runs.
  unset PYTHONHOME || true

  if [[ -n "${python_bin}" ]]; then
    if [[ ! -x "${python_bin}" ]]; then
      python_bin="$(command -v "${python_bin}" || true)"
    fi
    if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
      echo "Python executable is not available: ${PYTHON_BIN}" >&2
      exit 1
    fi

    local python_dir inferred_prefix
    python_dir="$(cd "$(dirname "${python_bin}")" && pwd -P)"
    python_bin="${python_dir}/$(basename "${python_bin}")"
    inferred_prefix="$(cd "${python_dir}/.." && pwd -P)"

    if [[ -n "${env_prefix}" ]]; then
      if [[ -d "${env_prefix}" ]]; then
        local resolved_env_prefix
        resolved_env_prefix="$(cd "${env_prefix}" && pwd -P)"
        if [[ "${resolved_env_prefix}" != "${inferred_prefix}" ]]; then
          echo "Warning: ENV_PREFIX and PYTHON_BIN point to different environments." >&2
          echo "  ENV_PREFIX=${resolved_env_prefix}" >&2
          echo "  PYTHON_BIN=${python_bin}" >&2
          echo "  Using the environment implied by PYTHON_BIN." >&2
        else
          env_prefix="${resolved_env_prefix}"
        fi
      else
        echo "Warning: ENV_PREFIX does not exist: ${env_prefix}" >&2
        echo "  Using the environment implied by PYTHON_BIN instead." >&2
      fi
    fi
    env_prefix="${inferred_prefix}"

    if [[ -f "${env_prefix}/bin/activate" ]]; then
      # shellcheck disable=SC1090
      source "${env_prefix}/bin/activate"
    fi
  elif [[ -n "${env_prefix}" && -d "${env_prefix}" ]]; then
    env_prefix="$(cd "${env_prefix}" && pwd -P)"
    export PATH="${env_prefix}/bin:${PATH}"
    if [[ -f "${env_prefix}/bin/activate" ]]; then
      # shellcheck disable=SC1090
      source "${env_prefix}/bin/activate"
    fi
    python_bin="${env_prefix}/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env_name}"
    python_bin="$(command -v python || true)"
    if [[ -n "${python_bin}" ]]; then
      local python_dir
      python_dir="$(cd "$(dirname "${python_bin}")" && pwd -P)"
      env_prefix="$(cd "${python_dir}/.." && pwd -P)"
    fi
  fi

  if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    echo "Could not find python. Set ENV_PREFIX or PYTHON_BIN explicitly." >&2
    exit 1
  fi
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="$(command -v "${python_bin}" || true)"
  fi
  if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
    echo "Python executable is not available: ${PYTHON_BIN:-${python_bin}}" >&2
    exit 1
  fi

  local python_dir
  python_dir="$(cd "$(dirname "${python_bin}")" && pwd -P)"
  python_bin="${python_dir}/$(basename "${python_bin}")"
  if [[ -z "${env_prefix}" ]]; then
    env_prefix="$(cd "${python_dir}/.." && pwd -P)"
  fi

  export PYTHON_BIN="${python_bin}"
  export ENV_PREFIX="${env_prefix}"
  export PATH="${env_prefix}/bin:${PATH}"
}
