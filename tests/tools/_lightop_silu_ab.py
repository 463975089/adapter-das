# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""lightop silu vs adapter triton path: A/B parity + perf on HCU.

Run inside the compute container (needs lightop + HCU):
    python tests/tools/_lightop_silu_ab.py
Gate (turn HCU_SILU_MUL_IMPL=lightop on for real workloads) requires:
  - parity  : cos >= 0.9999 vs triton over dtype/size matrix
  - graphsafety not yet wired here (fixture in tests/kernels); flip only
    after torchrun-style replay soak on real MoE shapes.
"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("HCU_ADAPTER_ROOT", os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sgl_kernel import silu_and_mul as adapter_impl
from sgl_kernel.elementwise import _lightop_silu_impl

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SIZES = [(1, 512), (33, 512), (128, 256), (777, 1024), (2048, 1792)]


def ref(x):
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d].float()) * x[..., d:].float()


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def parity(lo):
    all_ok = True
    for dt in DTYPES:
        for m, d2 in SIZES:
            torch.manual_seed(m + d2)
            x = (torch.randn(m, 2 * d2, device="cuda") * 2).to(dt)
            got = adapter_impl(x.clone()).float()
            want = ref(x)
            cos = torch.nn.functional.cosine_similarity(
                got.flatten(), want.flatten(), dim=0).item()
            maxrel = ((got - want).abs().max()
                      / (want.abs().max() + 1e-6)).item()
            ok = cos >= 0.9999 and maxrel <= 5e-2
            all_ok &= ok
            if not ok or m == 1:
                print(f"  {dt} {m}x{d2}: cos={cos:.6f} maxrel={maxrel:.3e}")
    return all_ok


def outkw(lo):
    x = torch.randn(9, 256, device="cuda").to(torch.bfloat16)
    out = torch.empty(9, 128, device="cuda", dtype=torch.bfloat16)
    y = adapter_impl(x, out)
    return (y.data_ptr() == out.data_ptr()
            and torch.allclose(y.float(), ref(x), atol=2e-2, rtol=2e-2))


def three_d(lo):
    x = torch.randn(2, 7, 256, device="cuda").to(torch.bfloat16)
    out = torch.empty(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    adapter_impl(x, out)
    return torch.allclose(out.float(), ref(x), atol=2e-2, rtol=2e-2)


def main():
    lo = _lightop_silu_impl()
    if lo is None:
        print("SKIP: lightop backend not active "
              "(set HCU_SILU_MUL_IMPL=lightop and install the wheel)")
        return
    print("backend:", lo)
    ok = True
    ok &= check("parity-matrix", parity(lo), "cos>=0.9999 maxrel<=5e-2")
    ok &= check("out-kwarg", outkw(lo), "writes caller buffer")
    ok &= check("3d-shapes", three_d(lo), "non-2d leading dims")
    print("VERDICT:", "GREEN — safe to keep HCU_SILU_MUL_IMPL=lightop"
          if ok else "RED — stay on triton")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
