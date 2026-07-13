#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to train outside a Slurm allocation." >&2
    exit 2
fi

if ! scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -Fxq "$(hostname -s)"; then
    echo "Refusing to train on $(hostname -s), which is not an allocated compute node." >&2
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
if [[ "${TERM:-dumb}" == "dumb" ]]; then
    export TERM=xterm-256color
fi

ISAACLAB_DIR="${ISAACLAB_DIR:-${LUSTRE_ROOT}/isaaclab/native-2.3.2/IsaacLab}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"
NUM_ENVS="${NUM_ENVS:-64}"

echo "node=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "max_iterations=${MAX_ITERATIONS}"
echo "num_envs=${NUM_ENVS}"
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv,noheader
GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)"
if [[ -n "${GPU_PROCESSES}" ]]; then
    echo "GPU already has compute processes:" >&2
    echo "${GPU_PROCESSES}" >&2
    exit 5
fi

cd "${ISAACLAB_DIR}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/cartpole-${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S).log"
mkdir -p "${LOG_DIR}"

set -o pipefail
timeout --signal=TERM --kill-after=60s "${TRAIN_TIMEOUT:-30m}" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cartpole-v0 \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --headless 2>&1 | tee "${LOG_FILE}"

# Require both CUDA execution and a finite RSL-RL completion.  A direct Slurm
# executable step supplies the high NOFILE limit checked at the top of this
# script, so GPU Foundation or descriptor failures are never accepted.
for marker in \
    '[INFO][AppLauncher]: Using device: cuda:0' \
    'Environment device    : cuda:0' \
    'Training time:'; do
    if ! grep -Fq "${marker}" "${LOG_FILE}"; then
        echo "Missing required training marker: ${marker}" >&2
        exit 9
    fi
done
if grep -Eq \
    'Failed to create any GPU devices|GPU Foundation is not initialized|no suitable CUDA GPU was found|expected number of used file descriptors|Traceback \(most recent call last\)|CUDA error:|Segmentation fault' \
    "${LOG_FILE}"; then
    echo "Training log contains a fatal runtime error." >&2
    exit 9
fi

echo "training_log=${LOG_FILE}"
echo "ISAACLAB_CARTPOLE_TRAINING_OK"
