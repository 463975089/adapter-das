# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone check: fast_hadamard_transform stub vs slow torch reference."""
import torch
import sgl_kernel  # registers the meta-path stub

from fast_hadamard_transform import hadamard_transform

torch.manual_seed(0)
dev = "cuda:0"
for n in (128, 64):
    x = torch.randn(37, n, device=dev, dtype=torch.bfloat16)
    # reference: explicit unnormalized matrix, scale applied separately
    h = torch.ones(1, 1, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    ref = ((x.float() @ h) * (n**-0.5)).to(torch.bfloat16)
    out = hadamard_transform(x, scale=n**-0.5)
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs().max().item()
    print(f"n={n}: shape={tuple(out.shape)} maxerr={err:.5f}")
    assert err < 5e-2, err
print("PASS")
