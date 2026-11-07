# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Breakable prefill CUDA graph (BCG) gate — driver-layer HIP truth.

sglang main (2026-11) routes prefill capture through the breakable CUDA
graph, whose runtime probes stream-capture status via cuda-python
(``cuda.bindings.runtime`` -> CUDA driver API,
runner_backend_utils/breakable_cuda_graph/cuda_utils.py). HCU has no CUDA
driver: ``checkCudaErrors(rt.cudaStreamGetCaptureInfo(...))`` dies mid
capture with "CUDA error 35 (CUDA driver version is insufficient...)"
(bw47 graph/mla, 2026-11-05; dense/moe/mamba never reach the probe —
only the MLA absorb path hooks _wait_stream).

The platform seam (governance §0.5) answers CUDA at the model layer on
purpose; BCG is a *driver-layer* feature, so the synthesis keeps it
HIP-truthful and routes breakable prefill to the eager runner — exactly
the upstream auto-disable arm of capture_prefill_graph ("resolved
prefill.backend='disabled'"): non-draft prefill keeps running through
the EagerRunner, decode FULL graphs never touch cuda-python and stay
enabled (validated: graph dense/moe/mamba green on gfx936).
"""

import logging

LOG = logging.getLogger("hcu.adapter")

_MARK = "_hcu_prefill_bcg_gate"


def install():
    import torch

    if not torch.version.hip:
        return

    from sglang.srt.model_executor import model_runner as _mr
    from sglang.srt.model_executor.model_runner_components import (
        cuda_graph_setup as _cgs,
    )
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )

    orig = _cgs.capture_prefill_graph
    if getattr(orig, _MARK, False):
        return

    def _hcu_capture_prefill_graph(
        *, model_runner, eager_runner, force_for_draft_worker=False
    ):
        if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
            draft = model_runner.is_draft_worker
            LOG.warning(
                "HCU adapter: breakable prefill CUDA graph -> eager prefill"
                " (cuda-python stream-capture probe has no CUDA driver on"
                " HCU; decode FULL graphs unaffected)"
            )
            return _cgs.GraphCapture(
                runner=None if draft else eager_runner,
                memory_phase="draft_prefill" if draft else "prefill",
                memory_usage_gb=0,
                capture_time=0,
            )
        return orig(
            model_runner=model_runner,
            eager_runner=eager_runner,
            force_for_draft_worker=force_for_draft_worker,
        )

    _hcu_capture_prefill_graph.__dict__[_MARK] = True
    import hcu_adapter.registry as _reg

    # Validate registration before publishing either namespace. A rejected
    # registration must not leave an active but unreported driver gate.
    _reg.register(
        "cuda_graph",
        "prefill-bcg-driver-truth",
        target=(
            "seam:sglang.srt.model_executor."
            "model_runner_components.cuda_graph_setup.capture_prefill_graph"
        ),
        mechanism=_reg.MECH_SEAM,
        detail="breakable prefill -> eager (cuda-python needs CUDA driver)",
    )
    # both namespaces: cuda_graph_setup's intra-module call and the
    # from-import copy inside model_runner (init_prefill_cuda_graph).
    _cgs.capture_prefill_graph = _hcu_capture_prefill_graph
    _mr.capture_prefill_graph = _hcu_capture_prefill_graph
