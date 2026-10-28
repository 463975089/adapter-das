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


def test_conv1d_pair():
    from hcu_adapter.conv1d_fn_emu import conv1d_fn_emu
    from hcu_adapter.conv1d_update_emu import causal_conv1d_update_emu
    from hcu_adapter.fla_hcu.conv1d_triton import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    DIM, W = 2048, 4
    T1, T2 = 2000, 512
    wt = torch.randn(DIM, W, device=DEV, dtype=torch.bfloat16) * 0.2
    bias = torch.randn(DIM, device=DEV, dtype=torch.bfloat16) * 0.1
    cu = torch.tensor([0, T1, T1 + T2], device=DEV, dtype=torch.int32)
    idx = torch.arange(2, device=DEV, dtype=torch.int32)
    cis = torch.tensor([True, True], device=DEV)

    x_tok = torch.randn(T1 + T2, DIM, device=DEV, dtype=torch.bfloat16)
    cs_e = torch.randn(SLOTS, DIM, W - 1, device=DEV, dtype=torch.bfloat16) * 0.3
    cs_f = cs_e.clone()
    o_e = conv1d_fn_emu(x_tok, wt, bias, conv_states=cs_e, query_start_loc=cu,
                        seq_lens_cpu=[T1, T2], cache_indices=idx,
                        has_initial_state=cis)
    o_f = causal_conv1d_fn(x_tok.t(), wt, bias, cs_f, cu, [T1, T2],
                           cache_indices=idx, has_initial_state=cis,
                           activation="silu")
    o_f = o_f.t() if o_f.shape[0] == DIM else o_f
    ok = _check("conv1d_fn out", o_e, o_f)
    ok &= (cs_e[idx] == cs_f[idx]).all().item()
    print(f"{'PASS' if ok else 'FAIL'} conv1d_fn state write-back bit-exact")

    BS = 16
    xu = torch.randn(BS, DIM, device=DEV, dtype=torch.bfloat16)
    su_e = torch.randn(SLOTS, DIM, W - 1, device=DEV, dtype=torch.bfloat16) * 0.3
    su_f = su_e.clone()
    ui = torch.arange(BS, device=DEV, dtype=torch.int32)
    oe = causal_conv1d_update_emu(xu, su_e, wt, bias, activation="silu",
                                  conv_state_indices=ui)
    of = causal_conv1d_update(xu, su_f, wt, bias, activation="silu",
                              conv_state_indices=ui)
    ok2 = _check("conv1d_update out", oe, of)
    ok2 &= (su_e[ui] == su_f[ui]).all().item()
    print(f"{'PASS' if ok2 else 'FAIL'} conv1d_update state bit-exact")
    return ok and ok2


if __name__ == "__main__":
    r1 = test_kda_prefill_chunk()
    r2 = test_conv1d_pair()
    print("ALL-PASS" if (r1 and r2) else "GATE-RED")
