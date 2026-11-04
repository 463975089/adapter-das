# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Inventory the operator surfaces of the three backend sources on HCU:
our sgl_kernel (triton), vendor aiter (DTK), vendor lightop (DTK 0.6.0).

Lists public callables per module so the replacement-matrix intersection
can be decided from facts, not memory.
"""

import importlib
import json
import types


def surface(modname):
    try:
        m = importlib.import_module(modname)
    except Exception as e:
        return {"error": repr(e)[:160]}
    out = []
    for k in dir(m):
        if k.startswith("_"):
            continue
        v = getattr(m, k)
        if isinstance(v, types.ModuleType):
            continue
        if callable(v):
            try:
                mod = getattr(v, "__module__", "") or ""
                # keep only names actually defined in this package
                if mod.startswith(modname.split(".")[0]):
                    out.append(k)
            except Exception:
                out.append(k)
    return sorted(set(out))


report = {}
for name in [
    "aiter",
    "aiter.ops",
    "aiter.ops.triton",
    "lightop",
    "lightop.elementwise",
    "lightop.attention",
    "lightop.fused",
    "sgl_kernel",
]:
    report[name] = surface(name)

# our adapter's own op modules (the production triton tier)
import os, sys
sys.path.insert(0, "/root/hcu_t2")
for sub in ["elementwise", "layernorm", "rope", "moe", "quant", "activation"]:
    try:
        report[f"sgl_kernel/{sub}"] = surface(f"sgl_kernel.{sub}")
    except Exception:
        pass

print(json.dumps(report, indent=0, default=str))
