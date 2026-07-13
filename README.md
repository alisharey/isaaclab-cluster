# Native Isaac Sim and Isaac Lab on MBZUAI HPC CLUSTER

This repository installs and validates Isaac Sim and Isaac Lab natively in a
per-user Conda environment. It is designed for this cluster's Slurm compute
nodes and does not require Docker, NVIDIA Container Toolkit, Apptainer,
Singularity, root access, or a local CUDA toolkit.

The validated stack is:

- Isaac Lab `v2.3.2`
- Isaac Sim `5.1.0.0`
- Python `3.11`
- PyTorch `2.7.0+cu128`
- torchvision `0.22.0+cu128`
- RSL-RL `3.1.2`

## What is stored where

Keep the small repository in home and the large environment, source checkout,
caches, and checkpoints on Lustre:

| Item | Default per-user location |
| --- | --- |
| This repository | wherever it was cloned, normally `$HOME/isaaclab-cluster` |
| Conda environment | name `isaaclab-2.3.2`, stored at `/l/users/$USER/conda-envs/isaaclab-2.3.2` |
| Isaac Lab checkout | `/l/users/$USER/isaaclab/native-2.3.2/IsaacLab` |
| Isaac/RTX cache | `/l/users/$USER/.cache/isaaclab-2.3.2` |
| Isaac configuration | `/l/users/$USER/.config/isaaclab-2.3.2` |
| Isaac runtime data | `/l/users/$USER/.local/share/isaaclab-2.3.2` |
| Wrapper logs and saved MP4s | `logs/` and `videos/` in this repository |
| RSL-RL checkpoints | `logs/rsl_rl/` inside the Lustre Isaac Lab checkout |

The scripts derive all paths from `$USER` and their own repository directory.
They contain no username, fixed job ID, or fixed compute-node dependency.

Optional overrides are available for unusual accounts or future cluster
changes:

```bash
export ISAACLAB_LUSTRE_ROOT=/l/users/$USER
export ISAACLAB_CONDA_ENV_NAME=isaaclab-2.3.2
export ISAACLAB_CONDA_ENVS_DIR="$ISAACLAB_LUSTRE_ROOT/conda-envs"
export ISAACLAB_DIR="$ISAACLAB_LUSTRE_ROOT/isaaclab/native-2.3.2/IsaacLab"
export ISAACLAB_CONDA_SH=/apps/local/anaconda2023/etc/profile.d/conda.sh
export ISAACLAB_LUSTRE_MOUNT=/l
```

`ISAACLAB_CONDA_PREFIX=/some/absolute/path` remains available as an advanced
override. Setting it deliberately switches creation and activation back to
prefix mode instead of the normal named-environment workflow.

## 1. Clone the repository

Clone or copy the repository into the user's home directory, then set one
convenience variable in the login shell:

```bash
cd "$HOME"
git clone REPOSITORY_URL isaaclab-cluster
export REPO_DIR="$HOME/isaaclab-cluster"
```

Replace `REPOSITORY_URL` with the URL where this directory is published. If it
was cloned elsewhere, set `REPO_DIR` to that absolute path.

## 2. Allocate distinct, unused GPU nodes

Choose any idle GPU node; the validated nodes are examples, not requirements.
One allocation is sufficient for installation and testing. If using the
account's allowance of two allocations, select two different nodes so the jobs
cannot share a device.

In separate login terminals:

```bash
NODE_A=replace-with-an-idle-gpu-node
salloc -w "$NODE_A" -n12 --mem=50G
```

Optionally allocate a second, different node:

```bash
NODE_B=replace-with-a-different-idle-gpu-node
salloc -w "$NODE_B" -n12 --mem=50G
```

From another login shell, record the job-to-node mapping:

```bash
squeue --me --noheader --format='job=%i node=%N state=%T'
export JOB_ID=replace-with-the-job-to-use
```

Verify that the selected GPU is unused before installation and before every
runtime launch:

```bash
srun --overlap --jobid="$JOB_ID" /usr/bin/nvidia-smi
srun --overlap --jobid="$JOB_ID" /usr/bin/nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader
```

The compute-process query must print no rows. Roughly 185-200 MiB attributed to
the node's display/Xorg service in the first command is normal. If a compute
PID appears, release that allocation and select another node. These CPU/memory
allocations do not reserve the GPU as a Slurm GRES, so the supplied wrappers
repeat the compute-process check and refuse to share an occupied GPU.

## 3. Attach safely when an interactive shell is needed

Use an explicit job ID when more than one allocation exists:

```bash
srun --pty --overlap --jobid="$JOB_ID" /bin/bash --noprofile --norc
```

This shell is useful for inspection, but launch the supplied runtime scripts
as direct `srun` executable steps as shown below. Do not use `bash -lc`, and do
not source `~/.bashrc` before an Isaac Sim launch.

An alias such as `attach_me` that selects the first queued job is ambiguous
when two jobs exist. A normal interactive Bash also reads the cluster startup
files, which lower the open-file hard limit to 2048. Always choose the job ID
explicitly for Isaac work.

## 4. Install the native environment

Run the installer directly in the allocation from the login shell:

```bash
srun --overlap --jobid="$JOB_ID" "$REPO_DIR/setup_native.sh"
```

The installer:

- refuses to run outside a Slurm allocation or on an occupied GPU;
- checks that at least 50 GiB is available on the per-user Lustre path;
- keeps transient Conda and pip archives in node-local `$TMPDIR`;
- registers `/l/users/$USER/conda-envs` as the first per-user Conda
  `envs_dirs` location;
- configures the concise `(environment-name)` prompt when the user has not
  already chosen a custom Conda prompt format;
- creates the named `isaaclab-2.3.2` Python 3.11 environment on Lustre;
- installs Isaac Sim 5.1 and CUDA 12.8 PyTorch wheels;
- clones the exact Isaac Lab `v2.3.2` tag on Lustre;
- installs the six Isaac Lab packages editably plus the RSL-RL backend;
- verifies versions, Git state, and Python dependency metadata; and
- writes `requirements-native-2.3.2.lock.txt` as an environment snapshot.

The script is idempotent. It does not invoke Kit or accept the NVIDIA EULA
during installation.

After installation, the environment has a normal Conda name and can be used
from any new shell without supplying its absolute prefix:

```bash
source /apps/local/anaconda2023/etc/profile.d/conda.sh
conda activate isaaclab-2.3.2
conda env list
```

For an older checkout created by a previous version of this repository with
`--prefix`, no environment copy or reinstall is necessary. Its directory
already has the correct basename. Register the parent once and Conda will show
and resolve it by name:

```bash
source /apps/local/anaconda2023/etc/profile.d/conda.sh
conda config --prepend envs_dirs "/l/users/$USER/conda-envs"
conda config --set env_prompt '({name}) '
conda activate isaaclab-2.3.2
```

Verify Python, package metadata, and a real CUDA tensor operation:

```bash
srun --overlap --jobid="$JOB_ID" "$REPO_DIR/verify_native_install.sh"
```

Success ends with `NATIVE_ENVIRONMENT_OK` and writes a log under
`$REPO_DIR/logs/`.

### Dependency choices

Isaac Sim 5.1 pins several older runtime packages. The repository constraints
keep those pins while satisfying Isaac Lab:

- `wheel==0.45.1`
- `setuptools==80.9.0`
- `flatdict==4.0.1` built without isolation
- `ipython==8.37.0`
- `psutil==5.9.8`
- `typing_extensions==4.12.2`
- `onnx==1.21.0`

One upstream metadata conflict cannot be simultaneously satisfied: Isaac Lab
0.54.2 declares `starlette==0.49.1`, while Isaac Sim's FastAPI 0.115.7 requires
`starlette<0.46`. This headless setup preserves simulator-compatible
`starlette==0.45.3`. `check_pip_dependencies.py` permits only that exact known
line and fails on any additional dependency conflict.

The upstream `isaaclab.sh --install` path also generates editor settings and
can import Isaac Sim during setup. `setup_native.sh` installs the equivalent
editable packages directly so installation remains noninteractive.

## 5. Accept the NVIDIA EULA

Each user must read and accept the
[NVIDIA Omniverse License Agreement](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/NVIDIA_Omniverse_License_Agreement.html)
before their first launch. Acceptance is passed only to the individual direct
step:

```bash
srun --overlap --jobid="$JOB_ID" /usr/bin/env \
  OMNI_KIT_ACCEPT_EULA=YES \
  "$REPO_DIR/run_headless_smoke.sh"
```

Success ends with `ISAAC_SIM_HEADLESS_OK`.

## 6. Run finite headless training

The default proof trains 64 Cartpole environments for two iterations:

```bash
srun --overlap --jobid="$JOB_ID" /usr/bin/env \
  OMNI_KIT_ACCEPT_EULA=YES \
  "$REPO_DIR/train_cartpole_headless.sh"
```

Success ends with `ISAACLAB_CARTPOLE_TRAINING_OK`. For a larger finite run:

```bash
srun --overlap --jobid="$JOB_ID" /usr/bin/env \
  OMNI_KIT_ACCEPT_EULA=YES \
  MAX_ITERATIONS=10 \
  NUM_ENVS=4096 \
  "$REPO_DIR/train_cartpole_headless.sh"
```

Wrapper logs go to `$REPO_DIR/logs/`; RSL-RL checkpoints and configuration
snapshots remain in the Lustre Isaac Lab checkout.

## 7. Save headless videos

Headless means there is no interactive window; it does not prevent off-screen
RTX rendering or MP4 recording. The video wrapper uses one Cartpole environment
and a fixed side view centered on environment 0 so the verification clip is not
obscured by a grid of cloned robots:

```bash
srun --overlap --jobid="$JOB_ID" /usr/bin/env \
  OMNI_KIT_ACCEPT_EULA=YES \
  "$REPO_DIR/train_cartpole_video.sh"
```

Defaults are 12 iterations, 64 frames per clip, and a 64-step recording
interval. Override `MAX_ITERATIONS`, `NUM_ENVS`, `VIDEO_LENGTH`, or
`VIDEO_INTERVAL` in the same `/usr/bin/env` command if needed. New MP4s are
copied into `$REPO_DIR/videos/`; the wrapper checks that at least one nonempty
file was created and uses `ffprobe` when available.

Success ends with `ISAACLAB_CARTPOLE_VIDEO_OK`.

## Why direct `srun` fixes the “open files” error

“Open files” here means per-process file descriptors used for plugins, shader
files, sockets, and runtime handles. It does not mean user documents are open,
damaged, or consuming storage.

On this cluster, an interactive Bash sources the Qlustar startup file, which
runs `ulimit -n 2048` and lowers both the soft and hard NOFILE limit. Isaac
Sim's GPU Foundation startup check expects 2450 descriptors for one GPU: a
2000 base plus 450 per selected GPU. A process whose hard limit is already 2048
cannot raise itself above that value.

A fresh direct Slurm executable step does not source Bash startup files and was
verified with soft/hard limits of `1048576/1048576`. The runtime wrappers print
both values and conservatively require at least 4096. Therefore:

- use direct `srun ... /usr/bin/env ... script.sh` for Kit workloads;
- do not wrap the command in `bash -lc`;
- do not launch from a shell that sourced `~/.bashrc`; and
- use absolute `/usr/bin/env` so a user-local executable cannot shadow it.

## Source-code modification policy

No upstream Isaac Lab source file is patched by this repository. The checkout
remains at tag `v2.3.2`, and the installer fails if tracked changes appear.
Editable installation means Python imports the clean checkout directly; it does
not imply that the source was modified.

All cluster-specific behavior lives in this repository's wrappers and in the
per-user Conda environment. The native workflow passed, so no container or
Singularity fallback was used.

## Expected warnings and troubleshooting

- `GLFW initialization failed` and `failed to open the default display` are
  expected when `DISPLAY` is intentionally unset for headless execution.
- The first rendered launch may spend around a minute compiling RTX shaders;
  later launches reuse the Lustre cache.
- If a wrapper reports an occupied GPU, choose a different allocation instead
  of sharing the device.
- If it reports NOFILE below 4096, return to the login shell and use the exact
  direct-step command from this guide.
- `/usr/local/cuda` is not a complete local toolkit here. The prebuilt Isaac
  Sim and PyTorch wheels do not need `nvcc`.
- Do not add old version-specific NVIDIA driver libraries to `LD_LIBRARY_PATH`;
  use the driver libraries selected by the node.
- A shared Isaac Sim 5.1 standalone ZIP exists under `/apps/local/isaac/`, but
  it is not used by this reproducible Conda/pip workflow.

## Validation record

Validated natively on 2026-07-13/14 on two distinct RTX 5000 Ada nodes with
driver 570.195.03 and CUDA driver API 12.8:

- Isaac Lab tag `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`;
- Python `3.11.15`, Isaac Sim `5.1.0.0`, PyTorch `2.7.0+cu128`;
- named activation `conda activate isaaclab-2.3.2` resolved to the expected
  per-user Lustre environment;
- CUDA tensor verification passed on `NVIDIA RTX 5000 Ada Generation`;
- native Vulkan/RTX startup selected GPU 0 and printed
  `ISAAC_SIM_HEADLESS_OK`;
- finite 64-environment RSL-RL training completed 2048 timesteps and printed
  `ISAACLAB_CARTPOLE_TRAINING_OK`;
- off-screen rendering produced three H.264 MP4 clips, each 1280x720, 60 fps,
  64 frames, and about 1.07 seconds long; and
- the upstream Isaac Lab checkout had no tracked or untracked modifications.

## Official references

- [Conda custom environment locations](https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/custom-env-and-pkg-locations.html)
- [Conda environment management and prompt names](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)
- [Isaac Lab native pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [Isaac Lab v2.3.2 compatibility information](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2#isaac-sim-version-dependency)
- [Isaac Lab cluster deployment guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html)
- [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Slurm `srun` documentation](https://slurm.schedmd.com/srun.html)
- [Slurm resource-limit FAQ](https://slurm.schedmd.com/faq.html#rlimits)
- [NVIDIA file-descriptor guidance](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/action_and_event_data_generation/tutorial_replicator_agent.html)
