#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to verify outside a Slurm allocation." >&2
    exit 2
fi

if ! scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -Fxq "$(hostname -s)"; then
    echo "Refusing to verify on $(hostname -s), which is not an allocated compute node." >&2
    exit 2
fi

LUSTRE_ROOT="${ISAACLAB_LUSTRE_ROOT:-/l/users/${USER}}"
ENV_PREFIX="${ISAACLAB_CONDA_PREFIX:-${LUSTRE_ROOT}/conda-envs/isaaclab-2.3.2}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${LUSTRE_ROOT}/isaaclab/native-2.3.2/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/native-verify-${SLURM_JOB_ID}-$(date +%Y%m%dT%H%M%S).log"

CONDA_SH="${ISAACLAB_CONDA_SH:-/apps/local/anaconda2023/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "Missing cluster Conda initialization: ${CONDA_SH}" >&2
    exit 6
fi
source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"
export PYTHONNOUSERSITE=1
mkdir -p "${SCRIPT_DIR}/logs"

set -o pipefail
{
    echo "node=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    python --version
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
    GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader)"
    if [[ -n "${GPU_PROCESSES}" ]]; then
        echo "GPU already has compute processes:" >&2
        echo "${GPU_PROCESSES}" >&2
        exit 5
    fi
    python "${SCRIPT_DIR}/check_pip_dependencies.py"
    python - <<'PY'
from importlib.metadata import version

import torch

print(f"isaacsim={version('isaacsim')}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print(f"torch_device={torch.cuda.get_device_name(0)}")
print(f"torch_capability={torch.cuda.get_device_capability(0)}")
x = torch.arange(1024, device="cuda", dtype=torch.float32)
result = float((x * x).sum().cpu())
assert result == 357389824.0, result
print(f"torch_cuda_sum={result}")
print("NATIVE_ENVIRONMENT_OK")
PY
    git -C "${ISAACLAB_DIR}" describe --tags --always
    git -C "${ISAACLAB_DIR}" rev-parse HEAD
} 2>&1 | tee "${LOG_FILE}"

echo "verify_log=${LOG_FILE}"
