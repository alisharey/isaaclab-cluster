#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to record outside a Slurm allocation." >&2
    exit 2
fi
if ! scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -Fxq "$(hostname -s)"; then
    echo "Refusing to record on $(hostname -s), which is not an allocated compute node." >&2
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
    echo "Open-file limit is too low for RTX video. Launch this executable as a direct srun step; do not use bash -lc or a shell that sourced ~/.bashrc." >&2
    exit 10
fi

CONDA_SH="${ISAACLAB_CONDA_SH:-/apps/local/anaconda2023/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "Missing cluster Conda initialization: ${CONDA_SH}" >&2
    exit 6
fi
source "${CONDA_SH}"
LUSTRE_ROOT="${ISAACLAB_LUSTRE_ROOT:-/l/users/${USER}}"
CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-isaaclab-2.3.2}"
CONDA_ENVS_DIR="${ISAACLAB_CONDA_ENVS_DIR:-${LUSTRE_ROOT}/conda-envs}"
ENV_PREFIX="${ISAACLAB_CONDA_PREFIX:-${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}}"
CONDA_ENV_TARGET="${ISAACLAB_CONDA_PREFIX:-${CONDA_ENV_NAME}}"
conda activate "${CONDA_ENV_TARGET}"
if [[ "${CONDA_PREFIX}" != "${ENV_PREFIX}" ]]; then
    echo "Conda resolved ${CONDA_ENV_TARGET} to ${CONDA_PREFIX}, expected ${ENV_PREFIX}." >&2
    exit 9
fi
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
MAX_ITERATIONS="${MAX_ITERATIONS:-12}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-64}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-64}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
SAVED_VIDEO_DIR="${SCRIPT_DIR}/videos"
RUN_STAMP="$(date +%Y%m%dT%H%M%S)"
RUN_STARTED="$(date +%s)"
LOG_FILE="${LOG_DIR}/cartpole-video-${SLURM_JOB_ID}-${RUN_STAMP}.log"
mkdir -p "${LOG_DIR}" "${SAVED_VIDEO_DIR}"

echo "node=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "conda_env_name=${CONDA_ENV_NAME}"
echo "conda_env_prefix=${CONDA_PREFIX}"
echo "max_iterations=${MAX_ITERATIONS}"
echo "num_envs=${NUM_ENVS}"
echo "video_length=${VIDEO_LENGTH}"
echo "video_interval=${VIDEO_INTERVAL}"
echo "camera_origin=env:0"
echo "camera_eye=7.0,0.0,2.5"
echo "camera_lookat=0.0,0.0,2.5"
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv,noheader
GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)"
if [[ -n "${GPU_PROCESSES}" ]]; then
    echo "GPU already has compute processes:" >&2
    echo "${GPU_PROCESSES}" >&2
    exit 5
fi

cd "${ISAACLAB_DIR}"
set -o pipefail
timeout --signal=TERM --kill-after=60s "${VIDEO_TIMEOUT:-30m}" \
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Cartpole-v0 \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --video \
        --video_length "${VIDEO_LENGTH}" \
        --video_interval "${VIDEO_INTERVAL}" \
        --headless \
        env.viewer.origin_type=env \
        env.viewer.env_index=0 \
        'env.viewer.eye=[7.0,0.0,2.5]' \
        'env.viewer.lookat=[0.0,0.0,2.5]' \
        2>&1 | tee "${LOG_FILE}"

if grep -Eq \
    'Failed to create any GPU devices|GPU Foundation is not initialized|no suitable CUDA GPU was found|Graphics plugins not available|expected number of used file descriptors|Traceback \(most recent call last\)|CUDA error:|Segmentation fault' \
    "${LOG_FILE}"; then
    echo "Video run reported a fatal GPU/rendering error." >&2
    exit 9
fi
for marker in \
    '[INFO][AppLauncher]: Using device: cuda:0' \
    '[INFO] Recording videos during training.' \
    'Training time:'; do
    if ! grep -Fq "${marker}" "${LOG_FILE}"; then
        echo "Missing required video marker: ${marker}" >&2
        exit 9
    fi
done

VIDEO_COUNT=0
while IFS= read -r video_file; do
    if (( $(stat -c %Y "${video_file}") < RUN_STARTED )); then
        continue
    fi
    saved_file="${SAVED_VIDEO_DIR}/cartpole-${SLURM_JOB_ID}-${RUN_STAMP}-$(basename "${video_file}")"
    cp --preserve=timestamps -- "${video_file}" "${saved_file}"
    [[ -s "${saved_file}" ]]
    echo "video_file=${saved_file}"
    if command -v ffprobe >/dev/null 2>&1; then
        ffprobe -v error -show_entries format=duration,size \
            -of default=noprint_wrappers=1 "${saved_file}"
    fi
    VIDEO_COUNT=$((VIDEO_COUNT + 1))
done < <(find "${ISAACLAB_DIR}/logs/rsl_rl/cartpole" -type f -name '*.mp4' -print)

if (( VIDEO_COUNT == 0 )); then
    echo "Training finished but no new MP4 was produced." >&2
    exit 11
fi

echo "video_count=${VIDEO_COUNT}"
echo "video_log=${LOG_FILE}"
echo "ISAACLAB_CARTPOLE_VIDEO_OK"
