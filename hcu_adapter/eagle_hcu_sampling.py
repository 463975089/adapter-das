# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Retain native HCU top-1 MTP sampling under the CUDA platform facade."""

import inspect

from . import registry

_installed = False


def _around_verify_policy(
    original, *, is_all_greedy, is_cpu, is_hip, is_xpu, use_rejection_sampling
):
    native = original(
        is_all_greedy=is_all_greedy, is_cpu=is_cpu, is_hip=is_hip,
        is_xpu=is_xpu, use_rejection_sampling=use_rejection_sampling,
    )
    if native or use_rejection_sampling:
        return native
    from sglang.srt.runtime_context import get_spec

    # Native eagle_sample has already sampled target IDs on HCU top-1 MTP.
    # Select its tree-matching path without replacing those samples.
    return get_spec().speculative_eagle_topk == 1


def install():
    global _installed
    if _installed:
        return True
    import torch

    if torch.version.hip is None:
        return False
    from sglang.srt.speculative import eagle_utils as eagle

    policy = getattr(eagle, "_verify_uses_greedy", None)
    if not getattr(eagle, "_is_hcu", False) or policy is None:
        return False
    if list(inspect.signature(policy).parameters) != [
        "is_all_greedy", "is_cpu", "is_hip", "is_xpu", "use_rejection_sampling",
    ] or not {"sample_mtp_target_ids", "_verify_uses_greedy"}.issubset(
        eagle.eagle_sample.__code__.co_names
    ):
        raise RuntimeError("Native HCU sampled-target verification contract changed")
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.speculative.eagle_utils._verify_uses_greedy"
    HookRegistry.register(target, _around_verify_policy, HookType.AROUND)
    registry.register(
        "platform_detect", "native HCU top-1 sampled-target verification",
        target, registry.MECH_SEAM,
    )
    _installed = True
    return True
