# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Keep the breakable graph's stream query on the HIP runtime."""

import logging


LOG = logging.getLogger("hcu.adapter")
_MARK = "_hcu_prefill_bcg_driver_truth"


def install():
    import torch

    from hcu_adapter.kill_switch import op_disabled

    if not torch.version.hip or op_disabled("prefill_graph_driver"):
        return

    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        breakable_cuda_graph as bcg,
    )

    if getattr(bcg.is_hip, _MARK, False):
        return

    from hcu_adapter import registry

    registry.register(
        "cuda_graph",
        "prefill-bcg-driver-truth",
        target=(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.breakable_cuda_graph.is_hip"
        ),
        mechanism=registry.MECH_REBIND,
        detail="stream capture status uses the HIP runtime on HCU",
    )

    def _hardware_is_hip():
        return torch.version.hip is not None

    setattr(_hardware_is_hip, _MARK, True)
    bcg.is_hip = _hardware_is_hip
    LOG.warning("HCU adapter: breakable graph stream queries use HIP runtime")
