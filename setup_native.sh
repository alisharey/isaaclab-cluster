#!/usr/bin/env bash

set -euo pipefail

ISAACLAB_TAG="v2.3.2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS_FILE="${SCRIPT_DIR}/native-constraints.txt"
LUSTRE_ROOT="${ISAACLAB_LUSTRE_ROOT:-/l/users/${USER}}"
LUSTRE_MOUNT="${ISAACLAB_LUSTRE_MOUNT:-/l}"
CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-isaaclab-2.3.2}"
CONDA_ENVS_DIR="${ISAACLAB_CONDA_ENVS_DIR:-${LUSTRE_ROOT}/conda-envs}"
ENV_PREFIX="${ISAACLAB_CONDA_PREFIX:-${CONDA_ENVS_DIR}/${CONDA_ENV_NAME}}"
CONDA_ENV_TARGET="${ISAACLAB_CONDA_PREFIX:-${CONDA_ENV_NAME}}"
ISAACLAB_DIR="${ISAACLAB_DIR:-${LUSTRE_ROOT}/isaaclab/native-2.3.2/IsaacLab}"
LOCAL_CACHE_ROOT="${TMPDIR:-/tmp}/${USER}/isaaclab-native-cache"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "Refusing to install outside a Slurm allocation." >&2
    exit 2
fi

if ! scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -Fxq "$(hostname -s)"; then
    echo "Refusing to install on $(hostname -s), which is not an allocated compute node." >&2
    exit 2
fi

echo "node=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader)"
if [[ -n "${GPU_PROCESSES}" ]]; then
    echo "GPU already has compute processes:" >&2
    echo "${GPU_PROCESSES}" >&2
    exit 5
fi

CONDA_SH="${ISAACLAB_CONDA_SH:-/apps/local/anaconda2023/etc/profile.d/conda.sh}"
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "Missing cluster Conda initialization: ${CONDA_SH}" >&2
    exit 6
fi
source "${CONDA_SH}"
conda --version

# Named environments are stored on Lustre rather than in the home directory.
# Registering their parent makes `conda activate <name>` work in future shells.
# An explicit ISAACLAB_CONDA_PREFIX retains the legacy/custom prefix behavior.
if [[ -z "${ISAACLAB_CONDA_PREFIX:-}" ]]; then
    mkdir -p "${CONDA_ENVS_DIR}"
    if ! conda config --show envs_dirs | grep -Fxq "  - ${CONDA_ENVS_DIR}"; then
        conda config --prepend envs_dirs "${CONDA_ENVS_DIR}"
    fi
    if [[ -z "$(conda config --get env_prompt)" ]]; then
        conda config --set env_prompt '({name}) '
    fi
fi

# Keep transient package archives on node-local storage.  The Conda environment
# and Isaac Lab checkout themselves remain persistent on Lustre.
export CONDA_PKGS_DIRS="${LOCAL_CACHE_ROOT}/conda-pkgs"
export PIP_CACHE_DIR="${LOCAL_CACHE_ROOT}/pip"
export PYTHONNOUSERSITE=1
mkdir -p \
    "${CONDA_PKGS_DIRS}" \
    "${PIP_CACHE_DIR}" \
    "$(dirname "${ENV_PREFIX}")" \
    "$(dirname "${ISAACLAB_DIR}")"

AVAILABLE_KB="$(df -Pk "${LUSTRE_ROOT}" | awk 'NR == 2 {print $4}')"
if (( AVAILABLE_KB < 50 * 1024 * 1024 )); then
    echo "Less than the required 50 GiB is free on ${LUSTRE_ROOT}." >&2
    exit 7
fi
df -h "${LUSTRE_ROOT}"
command -v lfs >/dev/null && lfs quota -u "${USER}" "${LUSTRE_MOUNT}" || true

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    if [[ -n "${ISAACLAB_CONDA_PREFIX:-}" ]]; then
        conda create --prefix "${ENV_PREFIX}" --yes python=3.11
    else
        conda create --name "${CONDA_ENV_NAME}" --yes python=3.11
    fi
fi

conda activate "${CONDA_ENV_TARGET}"
if [[ "${CONDA_PREFIX}" != "${ENV_PREFIX}" ]]; then
    echo "Conda resolved ${CONDA_ENV_TARGET} to ${CONDA_PREFIX}, expected ${ENV_PREFIX}." >&2
    exit 9
fi
echo "conda_env_name=${CONDA_ENV_NAME}"
echo "conda_env_prefix=${CONDA_PREFIX}"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
python --version
python -m pip install --upgrade pip

# Official Isaac Lab v2.3.2 native-pip dependency pins.
python -m pip install "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim 5.1 pins packaging==23.0.  New Conda environments currently ship
# wheel 0.47.0, whose metadata requires packaging>=24.  Wheel 0.45.1 has no
# runtime packaging requirement and removes that exact conflict.
python -m pip install --constraint "${CONSTRAINTS_FILE}" wheel==0.45.1

if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
    git clone --depth 1 --branch "${ISAACLAB_TAG}" \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

if [[ "$(git -C "${ISAACLAB_DIR}" describe --tags --exact-match 2>/dev/null || true)" != "${ISAACLAB_TAG}" ]]; then
    echo "${ISAACLAB_DIR} exists but is not checked out at ${ISAACLAB_TAG}." >&2
    exit 3
fi

cd "${ISAACLAB_DIR}"
if [[ "${TERM:-dumb}" == "dumb" ]]; then
    export TERM=xterm-256color
fi

# flatdict 4.0.1 still imports pkg_resources while building.  Setuptools 82
# removed that compatibility module, and pip's isolated build therefore fails.
# Build the pinned dependency once with the final setuptools release that still
# provides pkg_resources; Isaac Lab then sees the satisfied dependency.
python -m pip install --constraint "${CONSTRAINTS_FILE}" setuptools==80.9.0
python -m pip install --constraint "${CONSTRAINTS_FILE}" \
    --no-build-isolation flatdict==4.0.1

# Keep Isaac Sim kernel's exact runtime pins while satisfying the broader
# Isaac Lab notebook and ONNX requirements.  IPython 9 requires psutil>=7,
# whereas IPython 8.37 still satisfies ipywidgets and does not force that
# incompatible upgrade.
python -m pip install --constraint "${CONSTRAINTS_FILE}" \
    ipython==8.37.0 \
    psutil==5.9.8 \
    typing_extensions==4.12.2 \
    onnx==1.21.0

# The upstream helper always generates VS Code settings after a native install.
# That generator imports Isaac Sim and starts the interactive EULA flow.  Install
# the same editable packages directly so environment creation never launches Kit.
for extension in "${ISAACLAB_DIR}"/source/*; do
    [[ -f "${extension}/setup.py" ]] || continue
    python -m pip install --constraint "${CONSTRAINTS_FILE}" \
        --editable "${extension}"
done
python -m pip install --constraint "${CONSTRAINTS_FILE}" \
    --editable "${ISAACLAB_DIR}/source/isaaclab_rl[rsl_rl]"

# Isaac Lab 0.54.2 pins starlette==0.49.1 for its optional livestream server,
# while Isaac Sim 5.1 pins FastAPI 0.115.7, which requires starlette<0.46.
# Preserve the simulator side for headless Kit; the exact, sole metadata mismatch
# is checked by check_pip_dependencies.py instead of suppressing pip check broadly.
python -m pip install starlette==0.45.3

python - <<'PY'
from importlib.metadata import version

expected = {
    "isaacsim": "5.1.0",
    "isaaclab": "0.54.2",
    "isaaclab_tasks": "0.11.12",
    "torch": "2.7.0",
    "torchvision": "0.22.0",
}
for package, prefix in expected.items():
    installed = version(package)
    assert installed.startswith(prefix), f"{package}={installed}, expected {prefix}.*"
    print(f"{package}={installed}")
PY

git describe --tags --always
git rev-parse HEAD
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Isaac Lab checkout has tracked modifications after installation." >&2
    git status --short
    exit 8
fi

python "${SCRIPT_DIR}/check_pip_dependencies.py"
python -m pip list --format=freeze | LC_ALL=C sort | tee "${SCRIPT_DIR}/requirements-native-2.3.2.lock.txt"

echo "NATIVE_INSTALL_COMPLETE"
