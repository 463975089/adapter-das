# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit HIP worker policy, with opt-in graph capture and no NVIDIA PDL."""

from functools import partial
import os
import sys

from . import hcu_env
from hcu_adapter import registry


def _configure_worker_kernels(allow_cuda_graph=False):
    import torch
    from deepinfer import envs

    if envs.DEEPINFER_BATCH_INVARIANT:
        raise NotImplementedError("HCU batch-invariant kernels are not qualified")
    if not envs.DEEPINFER_ENFORCE_EAGER and not allow_cuda_graph:
        raise NotImplementedError("HCU graph candidate requires worker_runtime.install(allow_cuda_graph=True)")
    # Upstream enables VMM segments in _init_torch. On this DTK build even a
    # one-element allocation then fails on an idle card (recorded A/B probe).
    torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    mode = "eager" if envs.DEEPINFER_ENFORCE_EAGER else "graph candidate"
    print(f"HCU adapter runtime: worker kernels -> HIP {mode}; PDL and expandable segments disabled", flush=True)


def _configure_worker_environment():
    """Apply upstream worker settings, then disable the broken RCCL CuMem path."""
    import deepinfer.worker.runtime_config as runtime_config

    runtime_config._hcu_upstream_configure_worker_environment()
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    print(
        "HCU adapter runtime: NCCL_CUMEM_ENABLE=0 "
        "(DTK2610 RCCL CuMem path disabled after VMFault probe)",
        flush=True,
    )


def _prepare_collective_engram_driver():
    """RCCL lookup needs no NVIDIA peer-driver visibility expansion.

    The upstream hook initializes libcuda to expose peer allocations to the
    native Engram implementation. Our GPU-table provider exchanges lookup
    results with RCCL instead; normal Worker initialization owns HIP setup.
    Guard the active provider so a future vendor switch cannot inherit this
    policy without qualifying its peer-memory requirements.
    """
    import deep_ep
    from hcu_adapter.kernels.deepep_dtk import EngramBuffer, GraphEngramBuffer

    if deep_ep.EngramBuffer not in (EngramBuffer, GraphEngramBuffer):
        raise RuntimeError("HCU driver policy requires the qualified RCCL Engram provider")
    print("HCU adapter runtime: RCCL Engram needs no NVIDIA driver exposure; HIP init deferred to Worker", flush=True)


def install_cpu_affinity(physical_device_count):
    """Share the inherited CPU partition between workers and grammar builders."""
    from deepinfer import envs, utils

    allowed_cpus = sorted(os.sched_getaffinity(0))

    def partition(physical_rank):
        if not 0 <= physical_rank < physical_device_count:
            raise ValueError("Physical GPU ordinal is outside the verified node inventory")
        start = len(allowed_cpus) * physical_rank // physical_device_count
        end = len(allowed_cpus) * (physical_rank + 1) // physical_device_count
        cpus = allowed_cpus[start:end]
        if not cpus:
            raise RuntimeError("The container CPU mask cannot accommodate one CPU per worker")
        return cpus

    def bind_cpu(physical_rank):
        cpus = partition(physical_rank)
        os.sched_setaffinity(0, cpus)
        print(f"HCU adapter runtime: GPU {physical_rank} CPU affinity={cpus}", flush=True)

    def get_idle_cpus(physical_rank):
        cpus = partition(physical_rank)
        count = envs.DEEPINFER_SO_BUILDER_CPUS
        if not 0 <= count <= len(cpus):
            raise RuntimeError(f"Grammar builder requires {count} CPUs; GPU {physical_rank} has {len(cpus)}")
        return cpus[:count]

    utils.bind_cpu = bind_cpu
    utils.get_idle_cpus = get_idle_cpus
    worker_base = sys.modules.get("deepinfer.worker.base")
    if worker_base is not None:
        worker_base.bind_cpu = bind_cpu
    engine = sys.modules.get("deepinfer.engine.llm_engine")
    if engine is not None:
        engine.get_idle_cpus = get_idle_cpus
    registry.register("worker.cpu_affinity", "hcu_adapter.bridges._deepinfer.worker_runtime.install_cpu_affinity",
                      "deepinfer.utils.bind_cpu/get_idle_cpus", "rebind")


def install(physical_device_count, allow_cuda_graph=False):
    """Install before importing worker.base; preserve the caller's CPU mask."""
    if "worker.runtime" in hcu_env.resolve("HCU_DISABLE_OPS"):
        raise RuntimeError("HCU_DISABLE_OPS disables the worker runtime candidate")
    if physical_device_count <= 0:
        raise ValueError("A verified positive physical device count is required")
    from deepinfer.worker import runtime_config

    if not hasattr(runtime_config, "_hcu_upstream_configure_worker_environment"):
        runtime_config._hcu_upstream_configure_worker_environment = (
            runtime_config.configure_worker_environment
        )
    runtime_config.configure_worker_environment = _configure_worker_environment
    runtime_config.expose_all_gpus_to_cuda_driver = _prepare_collective_engram_driver
    configure_kernels = partial(_configure_worker_kernels, allow_cuda_graph=allow_cuda_graph)
    runtime_config.configure_worker_kernels = configure_kernels
    # WorkerBase imports these functions by value. Rebind the already-loaded
    # module as well, while avoiding an eager import of worker.base here.
    worker_base = sys.modules.get("deepinfer.worker.base")
    if worker_base is not None:
        worker_base.configure_worker_environment = _configure_worker_environment
        worker_base.configure_worker_kernels = configure_kernels
    install_cpu_affinity(physical_device_count)
    registry.register("worker.runtime", "hcu_adapter.bridges._deepinfer.worker_runtime.install",
                      "worker.runtime_config.configure_worker_kernels/deepinfer.utils.bind_cpu", "rebind")
    registry.register("worker.driver_exposure", "hcu_adapter.bridges._deepinfer.worker_runtime._prepare_collective_engram_driver",
                      "deepinfer.worker.runtime_config.expose_all_gpus_to_cuda_driver", "rebind")
    from . import bootstrap

    bootstrap.publish_worker_context(physical_device_count, allow_cuda_graph)
