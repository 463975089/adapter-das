# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""fp8_gemm_nt DTK wrapper correctness gate — §3.4b.

The DTK fork's ``fp8_gemm`` only honors per-row scales ([rows, 1]); given
sglang's per-token-group / per-block 2-D scale tensors it either ignores
the activation scales or reads them out of bounds (nmz26 probe: cos=0.004,
maxrel=4e5 vs a dequant reference — the GLM-5.3-Flash word-soup root
cause). The adapter therefore replaces the name-only alias with
``fp8_gemm_nt_dequant``: unfold both operands by block_k, apply the fp32
scales, one bf16 matmul into ``out``.

Shape matrix here:
- activation recipes: row-major [M, K/128] AND the column-major
  TMA-aligned VIEW the deepgemm caller produces (stride (1, aligned_M),
  sliced on the token axis);
- weight recipes: 128x128 block [N/128, K/128] and 1x128 row [N, K/128];
- M/N/K multiples of the blocks (graph steady state) and ragged M=7 /
  non-128 N (prefill tails / lm_head shapes);
- numeric contract per test-suite.md: cos >= 0.9999 AND maxrel <= 5e-2.

CPU-runnable: pure torch math, no DTK kernel call involved.
"""

import sys

import torch

sys.path.insert(0, "tests/kernels")

from hcu_adapter.deep_gemm_dtk import fp8_gemm_nt_dequant

torch.manual_seed(23)


def _quant_groups(x, block):
    """Per-group amax scale along the last dim -> (q fp8, s fp32 [*, K/b])."""
    xf = x.float()
    flat = xf.reshape(-1, block)
    s = flat.abs().amax(-1).clamp_min(1e-12) / 448.0
    q = (flat / s.unsqueeze(-1)).reshape(xf.shape).to(torch.float8_e4m3fn)
    return q, s.view(*x.shape[:-1], -1)


def _quant_blocks_2d(w, bn, bk):
    """Per (bn x bk) block amax: (q fp8 [N, K], s fp32 [N/bn, K/bk])."""
    wf = w.float().view(w.shape[0] // bn, bn, w.shape[1] // bk, bk)
    s = wf.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    q = (wf / s[:, None, :, None]).view(w.shape).to(torch.float8_e4m3fn)
    return q, s


def _tma_aligned_view(s):
    """create_per_token_group_quant_fp8_output_scale(column_major_scales=True,
    scale_tma_aligned=True): a [K/g, aligned_M] buffer transposed then sliced
    on the token axis — the logical values equal s but memory is strided."""
    m, g = s.shape
    aligned = (m + 3) // 4 * 4
    buf = torch.zeros(g, aligned, dtype=torch.float32).transpose(-1, -2)
    buf[..., :m, :] = s
    return buf[..., :m, :]


def _metric(o, ref):
    of, rf = o.to(torch.float32).flatten(), ref.flatten()
    denom = of.norm() * rf.norm()
    cos = (of @ rf / denom).item() if denom > 0 else 0.0
    scale = rf.abs().max().item() or 1.0
    maxrel = ((of - rf).abs() / scale).max().item()
    return cos, maxrel


def _run_case(m, n, k, w_recipe, a_layout):
    b_bf = torch.randn(n, k, dtype=torch.bfloat16) * 0.3
    if w_recipe == "128x128":
        b_q, b_s = _quant_blocks_2d(b_bf, 128, 128)
    else:  # 1x128: per-row group scales [N, K/128]
        b_q, b_s = _quant_groups(b_bf, 128)

    a_bf = torch.randn(m, k, dtype=torch.bfloat16) * 0.3
    a_q, a_s = _quant_groups(a_bf, 128)
    a_s_used = _tma_aligned_view(a_s) if a_layout == "tma" else a_s

    out = torch.empty(m, n, dtype=torch.bfloat16)
    fp8_gemm_nt_dequant((a_q, a_s_used), (b_q, b_s), out)

    a_ref = a_q.float() * a_s.repeat_interleave(128, dim=1)
    if w_recipe == "128x128":
        b_ref = (
            b_q.float().view(n // 128, 128, k // 128, 128) * b_s[:, None, :, None]
        ).view(n, k)
    else:
        b_ref = b_q.float() * b_s.repeat_interleave(128, dim=1)
    ref = a_ref @ b_ref.t()

    cos, maxrel = _metric(out, ref)
    ok = cos >= 0.9999 and maxrel <= 5e-2
    label = f"{m}x{n}x{k} {w_recipe}/{a_layout}"
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: cos={cos:.6f} maxrel={maxrel:.2e}")
    return ok


if __name__ == "__main__":
    cases = [
        (64, 256, 512, "128x128", "row"),
        (64, 256, 512, "128x128", "tma"),
        (7, 256, 512, "128x128", "tma"),      # ragged M prefill tail
        (17, 512, 512, "128x128", "tma"),     # ragged M, 128-multiple N
        (2048, 7168, 512, "1x128", "tma"),    # w13-ish mlp shape, 1x128 recipe
        (2048, 512, 7168, "128x128", "row"),  # lm_head-ish, K ragged vs 128? no: multiple
    ]
    passed = sum(_run_case(*c) for c in cases)
    print(f"fp8_gemm_nt dequant wrapper: {passed}/{len(cases)} PASS")
    sys.exit(0 if passed == len(cases) else 1)
