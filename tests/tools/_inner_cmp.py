# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare L0/L1 internals: server (/tmp/hcu_inner.pt) vs reference."""
import torch
import torch.nn.functional as F

srv = torch.load("/tmp/hcu_inner.pt", map_location="cpu")
ref = torch.load("/adapter/tests/tools/_ref_inner.pt", map_location="cpu")

print("server keys:", sorted(srv.keys()))
print("ref keys:   ", sorted(ref.keys()))
for k in sorted(set(srv) & set(ref)):
    s, r = srv[k].float().flatten(), ref[k].float().flatten()
    if s.shape != r.shape:
        print(f"{k}: shape {s.shape} vs {r.shape}")
        continue
    cos = F.cosine_similarity(s, r, dim=0).item()
    print(f"{k}: cos={cos:.6f} norm_s={s.norm():.4f} norm_r={r.norm():.4f} "
          f"ratio={(s.norm() / (r.norm() + 1e-9)).item():.4f}")
