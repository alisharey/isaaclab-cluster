#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to run Isaac Sim outside a Slurm allocation." >&2
    exit 2
fi

if ! scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -Fxq "$(hostname -s)"; then
    echo "Refusing to run on $(hostname -s), which is not an allocated compute node." >&2
    exit 2
fi

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
    echo "Set OMNI_KIT_ACCEPT_EULA=YES only after accepting the NVIDIA Omniverse EULA." >&2
    exit 4
fi

NOFILE_SOFT="$(ulimit -Sn)"
NOFILE_HARD="$(ulimit -Hn)"
echo "nofile_soft=${NOFILE_SOFT}"
echo "nofile_hard=${NOFILE_HARD}"
if [[ "${NOFILE_SOFT}" != "unlimited" ]] && (( NOFILE_SOFT < 4096 )); then
    echo "Open-file limit is too low for Isaac Sim. Launch this executable as a direct srun step; do not use bash -lc or a shell that sourced ~/.bashrc." >&2
    exit 10
fi

CONDA_SH="${ISAACLAB_CONDA_SH:-/apps/local/anaconda2023/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "Missing cluster Conda initialization: ${CONDA_SH}" >&2
    exit 6
fi
source "${CONDA_SH}"
LUSTRE_ROOT="${ISAACLAB_LUSTRE_ROOT:-/l/users/${USER}}"
conda activate "${ISAACLAB_CONDA_PREFIX:-${LUSTRE_ROOT}/conda-envs/isaaclab-2.3.2}"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LUSTRE_ROOT}/.cache/isaaclab-2.3.2}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${LUSTRE_ROOT}/.config/isaaclab-2.3.2}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${LUSTRE_ROOT}/.local/share/isaaclab-2.3.2}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${XDG_CACHE_HOME}/nvidia/ComputeCache}"
mkdir -p "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${CUDA_CACHE_PATH}"
unset DISPLAY

echo "node=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv,noheader
GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)"
if [[ -n "${GPU_PROCESSES}" ]]; then
    echo "GPU already has compute processes:" >&2
    echo "${GPU_PROCESSES}" >&2
    exit 5
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/headless-smoke-${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S).log"
mkdir -p "${LOG_DIR}"

set -o pipefail
timeout --signal=TERM --kill-after=60s "${SMOKE_TIMEOUT:-20m}" \
    python "${SCRIPT_DIR}/smoke_isaacsim.py" 2>&1 | tee "${LOG_FILE}"

if grep -Eq \
    'Failed to create any GPU devices|GPU Foundation is not initialized|no suitable CUDA GPU was found|Graphics plugins not available|expected number of used file descriptors' \
    "${LOG_FILE}"; then
    echo "Isaac Sim reported a fatal GPU-initialization error; rejecting the smoke marker." >&2
    exit 9
fi
grep -Fq 'ISAAC_SIM_HEADLESS_OK' "${LOG_FILE}"
echo "smoke_log=${LOG_FILE}"
