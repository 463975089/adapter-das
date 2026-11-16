# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from . import registry


def install():
    import torch
    from sglang.srt.distributed.device_communicators import custom_all_reduce as module

    if torch.version.hip is None or not module._is_cuda:
        return
    if getattr(module.dispatch_custom_allreduce, "_hcu_aiter_dispatch", False):
        return

    def dispatch_custom_allreduce(group, device):
        if not module.get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
            raise RuntimeError("HCU AITER all-reduce is disabled")
        if (
            module.envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
            or module.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        ):
            raise RuntimeError("HCU AITER all-reduce does not implement the requested deterministic path")
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        return partial(
            CustomAllreduce,
            enable_register_for_capturing=not module.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get(),
        )

    registry.register(
        "allreduce", "aiter_dist",
        target=module.__name__ + ".dispatch_custom_allreduce",
        mechanism=registry.MECH_REBIND,
        detail="use the official HIP provider when SGLang uses CUDA platform semantics",
    )
    dispatch_custom_allreduce._hcu_aiter_dispatch = True
    module.dispatch_custom_allreduce = dispatch_custom_allreduce
