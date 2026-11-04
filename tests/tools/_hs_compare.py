# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare server per-layer hidden states vs reference dump.

Mode inspect: send request, print structure of hidden_states field.
Mode cmp: cosine/rel per layer entry vs _ref_hiddens.pt.
"""
import json
import sys
import urllib.request

import torch
import torch.nn.functional as F

URL = "http://127.0.0.1:30064/generate"
req = {
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
    "return_hidden_states": True,
}
r = urllib.request.Request(
    URL, json.dumps(req).encode(), {"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(r, timeout=120).read())

print("meta_info keys:", list(resp.get("meta_info", {}).keys()))
hs = None
for k in ("hidden_states",):
    if k in resp:
        hs = resp[k]
        print("top-level key", k, "type", type(hs))
if hs is None:
    mi = resp.get("meta_info", {})
    if "hidden_states" in mi:
        hs = mi["hidden_states"]
        print("meta_info.hidden_states type", type(hs))

if sys.argv[1:2] == ["inspect"]:
    def probe(o, path="root", depth=0):
        if depth > 4:
            return
        if isinstance(o, list):
            print(f"{path}: list len={len(o)}")
            if o:
                probe(o[0], path + "[0]", depth + 1)
        elif isinstance(o, dict):
            print(f"{path}: dict keys={list(o.keys())[:8]}")
        else:
            s = str(o)
            print(f"{path}: {type(o).__name__} {s[:80]}")
    probe(hs)
    sys.exit(0)

# cmp mode: hs expected list (per layer) of flat floats or nested
ref = torch.load("/adapter/tests/tools/_ref_hiddens.pt", map_location="cpu")
refl = ref["hiddens"]
print(f"ref entries: {len(refl)} ids={ref['ids']}")


def to_t(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32)


if isinstance(hs, list) and hs and isinstance(hs[0], list):
    # nested: maybe [layer][token] or [layer] flat
    lay = [to_t(e) for e in hs]
elif isinstance(hs, list):
    lay = [to_t(e) for e in hs]
else:
    raise SystemExit(f"unhandled hs structure {type(hs)}")

print(f"server entries: {len(lay)}")
H = refl[0].shape[-1]
print(f"ref H={H}")

n = min(len(lay), len(refl))
for i in range(n):
    s = lay[i].float().flatten()
    r_ = refl[i].float().flatten()
    if s.shape != r_.shape:
        # maybe server returns all tokens: take last-token chunk
        if s.numel() % r_.numel() == 0 and s.numel() > r_.numel():
            s = s.view(-1, r_.numel())[-1]
        else:
            print(f"entry {i}: shape mismatch {s.shape} vs {r_.shape}")
            continue
    cos = F.cosine_similarity(s, r_, dim=0).item()
    rel = ((s - r_).norm() / (r_.norm() + 1e-9)).item()
    flag = "  <== DIVERGE" if cos < 0.999 else ""
    print(f"entry {i:2d}: cos={cos:.6f} rel={rel:.4f} "
          f"norm_s={s.norm():.3f} norm_r={r_.norm():.3f}{flag}")
