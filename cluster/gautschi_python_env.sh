#!/bin/bash
# Gautschi-specific Python environment resolution.
# Sources CUDA and conda modules, then activates the sticky conda env.

gautschi_resolve_python_env() {
  local conda_env_name="${CONDA_ENV_NAME:-sticky}"
  local env_prefix="${ENV_PREFIX:-}"

  module load cuda 2>/dev/null || true
  module load conda 2>/dev/null || true

  if [[ -n "${env_prefix}" && -d "${env_prefix}" ]]; then
    # Prefix-based env (e.g. on scratch).
    conda activate "${env_prefix}"
    PYTHON_BIN="${env_prefix}/bin/python"
  else
    conda activate "${conda_env_name}"
    PYTHON_BIN="$(command -v python)"
  fi

  if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Could not find python after activating env '${conda_env_name}'." >&2
    echo "Run the setup steps first (see cluster/README or guide)." >&2
    exit 1
  fi

  export PYTHON_BIN
  echo "python_bin=${PYTHON_BIN}"
  "${PYTHON_BIN}" -c 'import sys; print(f"python {sys.version}")'
}
