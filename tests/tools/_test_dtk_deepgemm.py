#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DTK deepgemm kernel-semantics regression probe (EP bridge triage).

Run on a single GPU inside the DTK container:
    CUDA_VISIBLE_DEVICES=6 python3 _test_dtk_deepgemm.py

Encodes the kernel facts established while bringing DeepEP up on HCU
(2026-10-11, nmz26). Re-run after any deepgemm wheel upgrade — a change
in any PASS/FAIL verdict below changes what the EP bridge must patch:

  1. bf16 masked GEMM, native (G,N,K) weight  -> FAIL (garbage, ~6x ref mag)
  2. bf16 masked GEMM, pack_bf16_masked_weight-> PASS (HIPC BM64/BN256/BK128)
  3. bf16 masked, config MODE=1000 (asm)     -> RuntimeError on this arch
  4. bf16 contiguous, 2D input                -> SEGFAULT (kernel VMFault)
  5. bf16 contiguous, 3D input                -> FAIL (wrong numbers)
  6. fp8 contiguous, packed weight            -> demands a_scale [M]/[M,1]
     (row/tensor-wise scales; sglang uses 128-block scales -> mismatch)
"""
import sys

import torch

import deepgemm


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda"
    G, M, K, N = 4, 128, 1024, 512
    rc = 0

    def check(name, ok):
        nonlocal rc
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        rc |= 0 if ok else 1

    a = torch.randn(G, M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(G, N, K, device=dev, dtype=torch.bfloat16)
    masked_m = torch.tensor([128, 64, 1, 0], device=dev, dtype=torch.int32)
    ref = [
        (a[g, : int(masked_m[g])].float() @ b[g].t().float())
        if int(masked_m[g]) > 0
        else None
        for g in range(G)
    ]

    # 1. masked + native weight: the trap that produced token soup e2e
    d = torch.empty(G, M, N, device=dev, dtype=torch.bfloat16)
    deepgemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, M)
    native_diff = max(
        (d[g, : int(masked_m[g])].float() - ref[g]).abs().max().item()
        for g in range(G)
        if ref[g] is not None
    )
    check(f"masked native weight maxdiff={native_diff:.3f} (expect ~190=broken)", native_diff < 1.0)

    # 2. masked + packed weight: the working path the alias auto-packs for
    pb = deepgemm.pack_bf16_masked_weight(b)
    d2 = torch.empty(G, M, N, device=dev, dtype=torch.bfloat16)
    deepgemm.m_grouped_bf16_gemm_nt_masked(a, pb, d2, masked_m, M)
    packed_diff = max(
        (d2[g, : int(masked_m[g])].float() - ref[g]).abs().max().item()
        for g in range(G)
        if ref[g] is not None
    )
    check(f"masked packed weight maxdiff={packed_diff:.3f} (bf16 noise floor)", packed_diff < 1.0)

    # 3. asm mode availability
    try:
        deepgemm.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, M, config={"MODE": 1000})
        print("INFO  asm MODE=1000 ran (was RuntimeError on gfx pre-fix)")
    except RuntimeError as e:
        print(f"INFO  asm MODE=1000 unavailable: {str(e)[:60]}")

    # 4/5. contiguous layouts (normal-mode prefill path)
    m_indices = torch.repeat_interleave(
        torch.arange(G, device=dev, dtype=torch.int32), M
    )
    try:
        a2d = a.reshape(G * M, K).contiguous()
        d3 = torch.empty(G * M, N, device=dev, dtype=torch.bfloat16)
        deepgemm.m_grouped_bf16_gemm_nt_contiguous(a2d, b, d3, m_indices)
        print("INFO  contiguous 2D ran (was VMFault segfault)")
    except Exception as e:
        print(f"INFO  contiguous 2D: {type(e).__name__}: {str(e)[:60]}")

    # 6. fp8 contiguous scale semantics
    a8 = (torch.randn(G * M, K, device=dev) * 0.2).to(torch.float8_e4m3fn)
    a_scale = torch.ones(G * M, K // 128, device=dev, dtype=torch.float32)
    b8 = (torch.randn(G, N, K, device=dev) * 0.2).to(torch.float8_e4m3fn)
    b_scale = torch.ones(G, N, K // 128, device=dev, dtype=torch.float32)
    pb8 = deepgemm.pack_fp8_contiguous_weight(b8)
    dfp8 = torch.empty(G * M, N, device=dev, dtype=torch.bfloat16)
    try:
        deepgemm.m_grouped_fp8_gemm_nt_contiguous(
            (a8, a_scale), (pb8, b_scale), dfp8, m_indices
        )
        print("INFO  fp8 contiguous accepted block scales (was [M]/[M,1] demand)")
    except RuntimeError as e:
        print(f"INFO  fp8 contiguous scale demand: {str(e)[:80]}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
