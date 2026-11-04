# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect

for m, names in [
    ("aiter.moe", ["aiter_moe", "MoeQuantType", "MoeSolutionType",
                   "get_aiter_moe_config"]),
    ("aiter.fused_moe", ["fused_moe"]),
    ("aiter.ops.shuffle", ["moe_layout_shuffle_gemm2"]),
]:
    try:
        mod = importlib.import_module(m)
        print(m, {n: hasattr(mod, n) for n in names})
    except Exception as e:
        print(m, "IMPORT_FAIL", repr(e)[:100])

try:
    from aiter.moe import aiter_moe

    print("aiter_moe sig:", str(inspect.signature(aiter_moe))[:260])
except Exception as e:
    print("aiter_moe sig fail:", repr(e)[:120])

try:
    from aiter.fused_moe import fused_moe as fm

    print("fused_moe sig:", str(inspect.signature(fm))[:200])
except Exception as e:
    print("fused_moe fail:", repr(e)[:120])
