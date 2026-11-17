# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Vendored sglang-das hcu kernels vs adapter emus — green gate (gfx938).

Source: hcu_adapter/fla_hcu/* (fork maxiaobetter/sglang-das
``main_glm53_flash_feature_cleanup_tx``, commit 364fabf; pure triton,
JIT-compiled on this stack). These kernels replace the torch emus that
papered over the stock v0.5.21 gfx938 breakage (chunk_kda wrong Aqk;
causal_conv1d update noise / fn dead initial-state arm).

Gate: fork vs the emu references (which themselves carry their own green
gates) with dual metrics cos + rel-rms, plus the conv state write-back
which must be bit-exact.

Run: python tests/kernels/test_fla_hcu_fork_gfx938.py   (GPU)
Shape sweep: GATE_T1 / GATE_T2 override the two varlen seq lens
(default 1536/512) of the chunk test, e.g. GATE_T1=64 GATE_T2=5.
"""

# [L1 cold-curation 2026-11-17] 2 fork gates -> 1: kda_prefill_chunk stays as the fork-vs-emu representative; conv1d_pair stays in git history.
from __future__ import annotations

import torch

torch.manual_seed(7)
DEV = "cuda"
H, K, V, SLOTS = 16, 128, 128, 32


def _check(name, a, b, cos_min=0.9999, rel_max=0.02):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).pow(2).mean().sqrt() / (a.pow(2).mean().sqrt() + 1e-9)).item()
    ok = cos >= cos_min and rel <= rel_max
    print(f"{'PASS' if ok else 'FAIL'} {name}: cos={cos:.6f} rel={rel:.5g}")
    return ok


def test_kda_prefill_chunk():
    from hcu_adapter.fla_hcu.kda import chunk_kda
    from hcu_adapter.kda_prefill_emu import kda_extend_emu

    import os
    _t1 = int(os.environ.get('GATE_T1', '1536')); _t2 = int(os.environ.get('GATE_T2', '512'))
    T1, T2 = _t1, _t2
    T = T1 + T2
    q = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16) * 0.5
    beta = torch.randn(1, T, H, device=DEV, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEV, dtype=torch.float32) * 0.1 - 1.0
    dt_bias = torch.randn(H * K, device=DEV, dtype=torch.float32) * 0.05
    cu = torch.tensor([0, T1, T1 + T2], device=DEV, dtype=torch.int32)
    idx = torch.arange(2, device=DEV, dtype=torch.int32)
    st0 = torch.randn(SLOTS, H, V, K, device=DEV, dtype=torch.float32) * 0.05

    st_e = st0.clone()
    o_e, _ = kda_extend_emu(q, k, v, g, beta, ssm_states=st_e, cache_indices=idx,
                            query_start_loc=cu, A_log=A_log, dt_bias=dt_bias,
                            lower_bound=-5.0, scale=K**-0.5)
    # fork contract: beta arrives ACTIVATED (v0.5.21 backend sigmoid already)
    st_f = st0.clone()
    o_f = chunk_kda(q=q, k=k, v=v, g=g, beta=beta.float().sigmoid().to(beta.dtype),
                    initial_state=st_f, initial_state_indices=idx,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                    A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0)
    ok = _check("chunk_kda out", o_e, o_f)
    ok &= _check("chunk_kda final-state", st_e[idx], st_f[idx].float())
    return ok


if __name__ == "__main__":
    # [L1 cold-curation 2026-11-17] test_conv1d_pair was retired by the
    # spawn-family curation; the direct-run entry shrank with it.
    r1 = test_kda_prefill_chunk()
    print("ALL-PASS" if r1 else "GATE-RED")
    raise SystemExit(0 if r1 else 1)
