# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare the three rope paths on HCU for qwen3 (neox) params:
  A) fused HIP kernel path (fused_qk_rope_reshape_and_cache) -- production
  B) plain fallback kernel (sgl_kernel.rotary_embedding, DTK triton) --
     the path my CP guard falls into when fused arg is dropped
  C) torch reference (neox half-split rotate)

All three on identical inputs with t == t_slot (decode-shaped, where A is
valid). If B disagrees with A/C, the CP garbage root cause is the plain
rope kernel, not the CP machinery.
"""
import os
import sys

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "5")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

dev = "cuda"
dtype = torch.bfloat16
H, Hk, D, rotary_dim = 32, 4, 128, 128
base, max_pos = 10000.0, 4096
T = 7  # tokens


def torch_ref(positions, q, k):
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=dev).float() / rotary_dim))
    freqs = torch.outer(positions.float(), inv)  # (T, D/2)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (T,1,D/2) broadcast over heads

    def rot(x):
        x1, x2 = x[..., : rotary_dim // 2], x[..., rotary_dim // 2 : rotary_dim]
        return torch.cat(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1
        ).to(x.dtype)

    return rot(q.clone()), rot(k.clone())


def main():
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding

    rope = RotaryEmbedding(
        head_size=D,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_pos,
        base=base,
        is_neox_style=True,
        dtype=dtype,
    ).to(dev)

    positions = torch.tensor([0, 1, 2, 5, 9, 100, 333], device=dev, dtype=torch.int64)
    q = torch.randn(T, H, D, device=dev, dtype=dtype)
    k = torch.randn(T, Hk, D, device=dev, dtype=dtype)
    # production convention: rope consumes 2D (T, H*D)
    q2, k2 = q.reshape(T, H * D), k.reshape(T, Hk * D)

    # C: reference
    q_ref, k_ref = torch_ref(positions, q.clone(), k.clone())

    # B: plain fallback kernel (guard path) -- fused arg None
    q_b, k_b = rope(positions, q2.clone(), k2.clone(), fused_set_kv_buffer_arg=None)
    q_b, k_b = q_b.view(T, H, D), k_b.view(T, Hk, D)

    # A: fused HIP path -- FusedSetKVBufferArg dataclass (v0.5.21 API; the
    # pre-dataclass dict shape died at rope.py:213 fused_args.value, rerun3)
    from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg

    pool_k = torch.zeros(16, Hk * D, device=dev, dtype=dtype)
    pool_v = torch.zeros(16, Hk * D, device=dev, dtype=dtype)
    slots = torch.arange(T, device=dev, dtype=torch.int64)
    fused_arg = FusedSetKVBufferArg(
        value=torch.randn(T, Hk * D, device=dev, dtype=dtype),
        k_buffer=pool_k,
        v_buffer=pool_v,
        cache_loc=slots,
    )
    q_a, k_a = rope(positions, q2.clone(), k2.clone(), fused_set_kv_buffer_arg=fused_arg)
    q_a, k_a = q_a.view(T, H, D), k_a.view(T, Hk, D)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()

    r_b_ref_q, r_b_ref_k = rel(q_b, q_ref), rel(k_b, k_ref)
    r_a_ref_q, r_a_ref_k = rel(q_a, q_ref), rel(k_a, k_ref)
    print(f"fallback-vs-ref: q rel={r_b_ref_q:.4f} k rel={r_b_ref_k:.4f}")
    print(f"fused-vs-ref:    q rel={r_a_ref_q:.4f} k rel={r_a_ref_k:.4f}")
    ok = max(r_b_ref_q, r_b_ref_k, r_a_ref_q, r_a_ref_k) < 0.05
    print("RESULT:", "PASS rope paths agree" if ok else "FAIL rope divergence")
    if not ok:
        print("q fallback[0,0,:6]", q_b[0, 0, :6].float().tolist())
        print("q ref       [0,0,:6]", q_ref[0, 0, :6].float().tolist())
        print("q fused     [0,0,:6]", q_a[0, 0, :6].float().tolist())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
