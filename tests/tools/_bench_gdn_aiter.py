# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Performance grid: GDN aiter route vs the upstream triton fla kernels.

Non-graph wall time per call (sync + perf_counter), the two rebind faces:

- chunk prefill state recursion (fwd_h): H v-heads, K=V=128,
  T in {512, 2048, 8192}, B in {1, 4}
- packed decode: B in {1, 8, 64}, H=32

aiter side calls the same targets the bridge rebinds to (aiter CK
blockdim64 fwd_h / aiter triton packed decode), so numbers are the
bridge's steady-state win.
"""

import time

import torch

from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
import aiter.ops.fla as afla


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _prefill_inputs(bs, T, H, K=128, V=128):
    torch.manual_seed(5)
    k = torch.randn(bs, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.3
    v = torch.randn(bs, T, H, V, device="cuda", dtype=torch.bfloat16) * 0.3
    beta = torch.rand(bs, T, H, device="cuda") * 0.9 + 0.05
    g = -torch.rand(bs, T, H, device="cuda").exp() * 0.1
    gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=None)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=gc, beta=beta, cu_seqlens=None, chunk_size=64)
    pool = torch.randn(bs, H, V, K, device="cuda", dtype=torch.float32) * 0.2
    idx = torch.arange(bs, device="cuda", dtype=torch.int32)
    return k, w, u, gc, pool, idx


def main():
    torch.manual_seed(0)
    print("== fwd_h (prefill state recursion) ==")
    for bs, T, H in [(1, 512, 32), (4, 512, 32), (1, 2048, 32),
                     (4, 2048, 32), (1, 8192, 32)]:
        K = V = 128
        k, w, u, gc, p0, idx = _prefill_inputs(bs, T, H, K, V)
        p1 = p0.clone()
        t_tri = bench(lambda: chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=gc, gk=None,
            initial_state=p0, initial_state_indices=idx,
            save_new_value=True, cu_seqlens=None))
        t_ai = bench(lambda: afla.chunk_gated_delta_rule_fwd_sglang_hip_blockdim64(
            k=k, w=w, u=u, g=gc, gk=None,
            initial_state=p1, initial_state_indices=idx,
            output_final_state=True, chunk_size=64, save_new_value=True,
            cu_seqlens=None, transpose_state_layout=True))
        print(f"bs={bs} T={T:5d} H={H}: triton={t_tri:7.2f}ms "
              f"aiter={t_ai:7.2f}ms speedup={t_tri / t_ai:.2f}x")

    print("== packed decode ==")
    H = 32
    K = V = 128
    HV = H
    for bs in (1, 8, 64):
        torch.manual_seed(11)
        qkv_dim = 2 * H * K + HV * V
        mixed_qkv = torch.randn(bs, qkv_dim, device="cuda",
                                dtype=torch.bfloat16) * 0.5
        a = torch.randn(bs, HV, device="cuda") * 0.1
        b = torch.randn(bs, HV, device="cuda") * 0.1
        A_log = torch.randn(HV, device="cuda") * 0.1
        dt_bias = torch.randn(HV, device="cuda") * 0.1
        scale = K**-0.5
        slots = max(bs * 3, 64)
        pool = torch.randn(slots, HV, V, K, device="cuda",
                           dtype=torch.float32) * 0.3
        idx = torch.randperm(slots, device="cuda")[:bs].to(torch.int32)
        o0 = torch.empty(bs, 1, HV, V, device="cuda", dtype=torch.bfloat16)
        t_tri = bench(lambda: fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv, a, b, A_log, dt_bias, scale, pool, o0, idx,
            use_qk_l2norm_in_kernel=True))
        # aiter triton decode shares signature; only time it once more
        t_ai = bench(lambda: __import__(
            "aiter.ops.triton.fla.fused_recurrent",
            fromlist=["fused_recurrent_gated_delta_rule_packed_decode"]
        ).fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=pool, out=o0,
            ssm_state_indices=idx, use_qk_l2norm_in_kernel=True))
        print(f"bs={bs:3d}: triton={t_tri:7.3f}ms aiter={t_ai:7.3f}ms "
              f"speedup={t_tri / t_ai:.2f}x")


if __name__ == "__main__":
    main()
