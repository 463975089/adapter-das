# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Kernel gate v2: DTK aiter fused-MoE at REAL model shapes (Qwen3-30B-A3B
per-rank dims) vs vectorized torch reference.

A) aiter.fused_moe.fused_moe  bf16
B) aiter.moe.aiter_moe        W16A16 (moe_c)
C) aiter.moe.aiter_moe        FP8_W8A8 (+weight shuffle)
Run: HIP_VISIBLE_DEVICES=6 python _probe_aiter_moe3.py
"""

import sys

import torch

torch.manual_seed(11)
E, M, K, NI, TOPK = 64, 2048, 2048, 1536, 8  # NI=intermediate; w1 [E,2NI,K]
DEV = "cuda"
N1 = 2 * NI
fails = []


def ref_moe(x, w1, w2, ti, tw):
    x32, w132, w232 = x.float(), w1.float(), w2.float()
    out = torch.zeros_like(x32)
    for j in range(TOPK):
        ecol = ti[:, j]
        wj = tw[:, j]
        for e in range(E):
            mask = ecol == e
            if not bool(mask.any()):
                continue
            gu = x32[mask] @ w132[e].T
            gate, up = gu[:, :NI], gu[:, NI:]
            h = gate * torch.sigmoid(gate) * up
            y = h @ w232[e].T
            out[mask] += wj[mask, None] * y
    return out


x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.3
w1 = torch.randn(E, N1, K, device=DEV, dtype=torch.bfloat16) * 0.05
w2 = torch.randn(E, K, NI, device=DEV, dtype=torch.bfloat16) * 0.05
ti = torch.randint(0, E, (M, TOPK), device=DEV, dtype=torch.int32)
tw = torch.softmax(torch.randn(M, TOPK, device=DEV, dtype=torch.float32), -1)
print("shapes", x.shape, w1.shape, w2.shape, flush=True)
ref = ref_moe(x, w1, w2, ti, tw)
print("ref max", ref.abs().max().item(), flush=True)


def check(tag, out):
    md = (out.float() - ref).abs().max().item()
    rel = md / (ref.abs().max().item() + 1e-6)
    ok = rel < 0.05
    print(f"{tag}: maxdiff={md:.4f} rel={rel:.2e} {'OK' if ok else 'FAIL'}",
          flush=True)
    if not ok:
        fails.append(tag)


try:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    out = fused_moe(
        hidden_states=x, w1=w1, w2=w2,
        topk_weight=tw, topk_ids=ti,
        activation=ActivationType.Silu, quant_type=QuantType.No,
    )
    torch.cuda.synchronize()
    check("A-fused_moe-bf16", out)
except Exception:
    import traceback
    traceback.print_exc()
    fails.append("A-exception")

try:
    from aiter.moe import (MoeQuantType, MoeSolutionType, aiter_moe,
                           get_aiter_moe_config)
    from aiter.ops.shuffle import moe_layout_shuffle_gemm2

    def run_aiter_moe(tag, qt, w1x, w2x, w1s=None, w2s=None):
        status, cfg = get_aiter_moe_config(
            M=M, E=E, N1=N1, N2=K, K=K, top_k=TOPK, block_size=0,
            dtype=x.dtype, quant_type=qt, activation="silu")
        print(f"{tag}: status={status} solution={cfg.solution_type} "
              f"shuffle={getattr(cfg, 'need_shuffle', '?')}", flush=True)
        if not status:
            fails.append(tag + "-noconfig")
            return
        aw1, aw2 = w1x, w2x
        need = getattr(cfg, "need_shuffle",
                       cfg.solution_type == MoeSolutionType.MOE_C)
        if need:
            aw1 = moe_layout_shuffle_gemm2(w1x).view(w1x.shape)
            aw2 = moe_layout_shuffle_gemm2(w2x).view(w2x.shape)
        out = aiter_moe(
            hidden_states=x, w1=aw1, w2=aw2,
            topk_weights=tw.to(torch.float32), topk_ids=ti.to(torch.int32),
            moe_config=cfg, inplace=False, activation="silu",
            w1_scale=w1s, w2_scale=w2s, w1_zp=None, w2_zp=None,
            a1_scale=None, a2_scale=None, block_shape=None,
            global_num_experts=E, expert_map=None,
            routed_scaling_factor=1.0, output_dtype=x.dtype,
        )
        torch.cuda.synchronize()
        check(tag, out)

    run_aiter_moe("B-aiter_moe-w16a16", MoeQuantType.W16A16, w1, w2)

    def blk_quant(w):
        E_, N_, K_ = w.shape
        wv = w.float()
        nb_n, nb_k = (N_ + 127) // 128, (K_ + 127) // 128
        scale = torch.empty(E_, nb_n, nb_k, device=DEV, dtype=torch.float32)
        q = torch.empty_like(wv)
        for e in range(E_):
            for i in range(nb_n):
                for j in range(nb_k):
                    blk = wv[e, i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
                    s = blk.abs().max().clamp(min=1e-6) / 448.0
                    scale[e, i, j] = s
                    q[e, i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (
                        blk / s).clamp(-448, 448)
        return q.to(torch.float8_e4m3fn), scale

    w1q, w1s = blk_quant(w1)
    w2q, w2s = blk_quant(w2)
    run_aiter_moe("C-aiter_moe-fp8w8a8", MoeQuantType.FP8_W8A8, w1q, w2q,
                  w1s, w2s)
except Exception:
    import traceback
    traceback.print_exc()
    fails.append("BC-exception")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
