# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json

out = {}
try:
    import lightop
    out["file"] = lightop.__file__
    ver = getattr(lightop, "__version__", "?")
    out["version"] = str(ver)
    top = [n for n in dir(lightop) if not n.startswith("_")]
    out["top_names"] = top
except Exception as e:
    out["import_err"] = repr(e)[:300]

print(json.dumps(out, indent=1, default=str)[:3000])

# deep submodules relevant to P1
for mod in ("lightop.ops", "lightop.ops.triton", "lightop.triton",
            "lightop.fused", "lightop.attention"):
    try:
        m = __import__(mod, fromlist=["x"])
        names = [n for n in dir(m) if not n.startswith("_")]
        print(f"{mod}: {names[:40]}")
    except Exception as e:
        print(f"{mod}: ERR {repr(e)[:80]}")
