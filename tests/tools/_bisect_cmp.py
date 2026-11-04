# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Layer-by-layer parity bisect: server dump vs reference dump.

Server rows are (layer_idx, T, tensor); reference rows are plain tensors
(embed, L0..L_{NL-1}, final_normed). Entry i of reference = residual stream
after embed/layer i; server row for layer i = stream entering layer i's
attention = reference entry i.
"""
import torch
import torch.nn.functional as F

srv = torch.load("/tmp/hcu_layers.pt", map_location="cpu")
ref = torch.load("/adapter/tests/tools/_ref_hiddens.pt", map_location="cpu")["hiddens"]

if srv and isinstance(srv[0], tuple):
    print("server rows:", [(r[0], r[1]) for r in srv])
    lay = {r[0]: (r[1], r[2]) for r in srv}  # last write wins per layer
else:
    lay = {i: (None, r) for i, r in enumerate(srv)}

first_bad = None
for i in range(len(lay)):
    t_cnt, s = lay[i]
    s = s.float().flatten()
    if i >= len(ref):
        break
    r = ref[i].float().flatten()
    if s.shape != r.shape:
        print(f"L{i}: shape {s.shape} vs {r.shape}")
        continue
    cos = F.cosine_similarity(s, r, dim=0).item()
    rel = ((s - r).norm() / (r.norm() + 1e-9)).item()
    bad = cos < 0.999
    if bad and first_bad is None:
        first_bad = i
    print(f"L{i:<2d} (T={t_cnt}): cos={cos:.6f} rel={rel:.4f} "
          f"norm_s={s.norm():.3f} norm_r={r.norm():.3f}"
          f"{'  <== DIVERGE' if bad else ''}")
print("first divergent layer:", first_bad)
