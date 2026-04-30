#!/bin/bash
# Submit the OWT prep pipeline:
#   1. 5-task shard array (cpu/standby) — tokenize + stream to .bin files
#   2. dependent merge job — bins -> (N, 1024) int32 .npy + log_w.npz
#
# Both jobs land on cpu/standby (only QOS available to ruqiz on cpu).
# The merge runs only if every shard exits 0 (afterok). Shards are
# preempt-resilient via --requeue and the .tmp -> final rename in the
# patched prepare_openwebtext.py.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SHARDS_SBATCH=${SCRIPT_DIR}/prep_owt_shards.sbatch
MERGE_SBATCH=${SCRIPT_DIR}/prep_owt_merge.sbatch

if [[ ! -f "${SHARDS_SBATCH}" || ! -f "${MERGE_SBATCH}" ]]; then
  echo "ERROR: missing sbatch script(s) under ${SCRIPT_DIR}" >&2
  exit 1
fi

ARRAY_JOBID=$(sbatch --parsable "${SHARDS_SBATCH}")
echo "Shard array submitted: ${ARRAY_JOBID}"

MERGE_JOBID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOBID}" "${MERGE_SBATCH}")
echo "Merge submitted:       ${MERGE_JOBID}  (waits on afterok:${ARRAY_JOBID})"

echo ""
echo "=== Queue snapshot ==="
squeue -u "${USER}" -j "${ARRAY_JOBID},${MERGE_JOBID}"
