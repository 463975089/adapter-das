# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding (neox layout), GPU contract vs elementwise reference.

Layout regression fbc540c lived here: the cache split is half/half
([cos | sin]), not interleaved — an interleave shift still "runs" but
destroys cos. CPU-side interleaved layout is covered by
tests/test_ops_cpu.py::test_rope_interleaved_cpu.

Run on a node with a working GPU: `pytest tests/kernels/test_rope.py -v`.
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_neox_gpu(dtype):
    torch.manual_seed(5)
    T, qH, kvH, D, rot = 37, 8, 4, 128, 64
    cache = torch.randn(128, rot, device="cuda", dtype=torch.float32)
    pos = torch.randint(0, 128, (T,), device="cuda")
    q = torch.randn(T, qH * D, device="cuda").to(dtype)
    k = torch.randn(T, kvH * D, device="cuda").to(dtype)
    q0, k0 = q.clone(), k.clone()

    sgl_kernel.rotary_embedding(pos, q, k, D, cache, is_neox=True)

    for t in (0, 5, 36):
        p = pos[t].item()
        c = cache[p, : rot // 2]
        s = cache[p, rot // 2:]
        for base0, nH, t0 in ((q0, qH, q), (k0, kvH, k)):
            for h in range(nH):
                off = h * D
                x1 = base0[t, off: off + rot // 2].float()
                x2 = base0[t, off + rot // 2: off + rot].float()
                torch.testing.assert_close(
                    t0[t, off: off + rot // 2].float(), x1 * c - x2 * s,
                    rtol=2e-2, atol=2e-2,
                )
                torch.testing.assert_close(
                    t0[t, off + rot // 2: off + rot].float(), x1 * s + x2 * c,
                    rtol=2e-2, atol=2e-2,
                )
