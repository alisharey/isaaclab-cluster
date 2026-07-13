"""Finite native Isaac Sim headless smoke test."""

import os

from isaacsim import SimulationApp


cpu_threads = max(1, int(os.environ.get("ISAACLAB_CPU_THREADS", os.environ.get("SLURM_CPUS_ON_NODE", "12"))))


simulation_app = SimulationApp(
    {
        "headless": True,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "disable_viewport_updates": True,
        "limit_cpu_threads": cpu_threads,
    }
)
try:
    simulation_app.update()
    print("ISAAC_SIM_HEADLESS_OK", flush=True)
finally:
    simulation_app.close()
