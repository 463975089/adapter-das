# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Serve-time probe: which fp8 GEMM path + shapes does Qwen3-4B-Channel-FP8
actually take? Run server with HCU_FP8_TRACE=1 in env via this wrapper's
sitecustomize-level trace, then send one request and read /tmp/fp8_trace.log.
"""

import os
import sys

sys.path.insert(0, "/adapter")

import torch

# after sglang import (sitecustomize ran), wrap the apply functions
import sglang.srt.layers.quantization.fp8_utils as fu

LOG = open("/tmp/fp8_trace.log", "w", buffering=1)  # line-buffered

for name in (
    "apply_fp8_linear",
    "apply_w8a8_block_fp8_linear",
    "apply_fp8_ptpc_linear",
):
    fn = getattr(fu, name, None)
    if fn is None:
        continue

    def wrap(fn, name):
        def w(*a, **k):
            input = a[0] if a else k.get("input")
            weight = a[1] if len(a) > 1 else k.get("weight")
            extra = ""
            try:
                ws = a[2] if len(a) > 2 else k.get("weight_scale")
                extra = f" w{tuple(weight.shape)} ws{tuple(ws.shape)}[{ws.numel()}]"
                if isinstance(input, torch.Tensor):
                    extra += f" x{tuple(input.shape)}.{str(input.dtype).split('.')[-1]}"
            except Exception:
                pass
            try:
                out = fn(*a, **k)
                LOG.write(f"{name} OK x? {extra}\n")
                return out
            except Exception as e:
                LOG.write(f"{name} RAISE {e!r:.120} {extra}\n")
                raise

        return w

    import sglang.srt.layers.quantization.fp8_utils as _m

    setattr(_m, name, wrap(fn, name))

print("[fp8-trace] installed", flush=True)
