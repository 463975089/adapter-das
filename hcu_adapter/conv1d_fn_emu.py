"""causal_conv1d_fn torch emulator — gfx938 stopgap (prefill varlen arm).

The DTK triton `_causal_conv1d_fwd_kernel` miscompiles on gfx938: with
``has_initial_state`` set the initial-state load/store branches produce noise
(parallel to the confirmed ``causal_conv1d_update`` bug — red gate
``tests/kernels/test_conv1d_fn_varlen.py`` holds it: out cos 0.49 vs a
per-token step reference, state write-back maxdiff > 2). All 42 KDA layers
of GLM-5.3-Flash route their prefill through this kernel, so the logits are
noise from token one.

This module replaces ``causal_conv1d_fn`` with a sequential per-request
torch transcription of the documented contract:

    for each request b (slot = cache_indices[b] if given else b):
        win  = concat(state[slot] cache line b, x tokens t0..t1)
        out[:, t] = silu(w[row] . win[:, i:i+W] + bias[row])
        state[slot] <-- last W-1 tokens of (state-prefix + x)   [write-back]

`has_initial_state[b]`=False treats the pre-window as zeros (fresh request);
x full channels [dim, cu_seqlen] (channel-last strides as the launcher
asserts) and [T, dim] both accepted. Selected via HCU_CONV1D_IMPL=emu
(hip default; =fork routes to the vendored fixed kernel — see
fla_hcu_router). Sequential in T — correctness
path, not a production-throughput kernel.
"""

from __future__ import annotations

import os

import torch


def conv1d_fn_emu(
    x: torch.Tensor,                      # [dim, cu_seqlen] or [T, dim]
    weight: torch.Tensor,                 # [dim, W]
    bias: torch.Tensor | None,            # [dim]
    conv_states: torch.Tensor | None,     # [lines, dim, W-1]
    query_start_loc: torch.Tensor,        # [B+1] int32
    seq_lens_cpu,                         # list[int] | tensor (batch)
    cache_indices: torch.Tensor | None,   # [B] int32
    has_initial_state: torch.Tensor | None,  # [B] bool
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    **_ignored,
):
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (got {x.shape})")
    dim = weight.shape[0]
    W = weight.shape[1]
    state_len = W - 1
    B = query_start_loc.numel() - 1
    device = weight.device

    if os.environ.get("HCU_TRACE_CONVFN_EMU") == "1":
        print(f"[hcu conv1d_fn emu] call shapes x={tuple(x.shape)} W={W} B={B}",
              flush=True)
    # canonical tokens-major view [T, dim]; the launcher contract is
    # channel-last x [dim, cu_seqlen] with stride(0)==1, but accept [T, dim]
    # (token-major, stride(1)==1) too — kda_backend transposes both ways.
    if x.shape[0] == dim:
        x_tok = x.t()
    else:
        x_tok = x
    assert x_tok.shape[1] == dim, (x_tok.shape, dim)
    xt_f = x_tok.float()

    lens = [int(v) for v in list(seq_lens_cpu)] if not torch.is_tensor(seq_lens_cpu) else [int(v) for v in seq_lens_cpu.tolist()]
    assert sum(lens) == x_tok.shape[0], (lens, x_tok.shape)

    out = torch.empty_like(x_tok)
    act = activation in ("silu", "swish") or activation is True
    w_f = weight.float()
    b_f = bias.float() if bias is not None else None

    for bidx in range(B):
        t0 = int(query_start_loc[bidx])
        t1 = t0 + lens[bidx]
        if cache_indices is not None:
            slot = int(cache_indices[bidx])
            if slot == pad_slot_id:
                continue
        else:
            slot = bidx
        init = bool(has_initial_state[bidx]) if has_initial_state is not None else False
        if conv_states is not None:
            hist = conv_states[slot].float().clone()  # [dim, W-1]
            if not init:
                hist.zero_()
        else:
            hist = torch.zeros(dim, state_len, device=device)
        hist = torch.cat([hist, torch.zeros(dim, t1 - t0, device=device)], dim=1)
        L = t1 - t0
        hist[:, state_len:state_len + L] = xt_f[t0:t1].t()
        # windowed dot: [dim, L] each col i uses hist cols i..i+W-1
        win = hist.unfold(1, W, 1)               # [dim, L, W]
        val = (win * w_f.unsqueeze(1)).sum(-1)   # [dim, L]
        if b_f is not None:
            val = val + b_f.unsqueeze(1)
        if act:
            val = torch.nn.functional.silu(val)
        out[t0:t1] = val.t().to(out.dtype)
        if conv_states is not None:
            conv_states[slot] = hist[:, -(state_len):].to(conv_states.dtype)
    # return in the caller's layout: same shape as x (caller transposes both
    # ways). Contiguous copy — the triton kernel returned a fresh contiguous
    # tensor, and downstream graph-capture views/matmuls assume those strides.
    if out.shape != x.shape:
        out = out.t()
    return out.contiguous()


def install_patch():
    from sglang.kernels.ops.mamba import causal_conv1d_triton as m

    if getattr(m, "_hcu_convfn_emu_patched", False):
        return
    _orig = m.causal_conv1d_fn

    def causal_conv1d_fn(x, weight, bias=None, conv_states=None,
                         query_start_loc=None, seq_lens_cpu=None,
                         cache_indices=None, has_initial_state=None,
                         activation=None, pad_slot_id=-1, **kwargs):
        return conv1d_fn_emu(
            x, weight, bias, conv_states, query_start_loc, seq_lens_cpu,
            cache_indices, has_initial_state, activation, pad_slot_id,
            **kwargs,
        )

    m.causal_conv1d_fn = causal_conv1d_fn
    m._hcu_convfn_emu_patched = True
    m._hcu_convfn_emu_orig = _orig
    from sglang.kernels.ops.mamba import causal_conv1d_triton as m

    if getattr(m, "_hcu_convfn_emu_patched", False):
        return
    _orig = m.causal_conv1d_fn

    def causal_conv1d_fn(x, weight, bias=None, conv_states=None,
                         query_start_loc=None, seq_lens_cpu=None,
                         cache_indices=None, has_initial_state=None,
                         activation=None, pad_slot_id=-1, **kwargs):
        return conv1d_fn_emu(
            x, weight, bias, conv_states, query_start_loc, seq_lens_cpu,
            cache_indices, has_initial_state, activation, pad_slot_id,
            **kwargs,
        )

    m.causal_conv1d_fn = causal_conv1d_fn
    m._hcu_convfn_emu_patched = True
    m._hcu_convfn_emu_orig = _orig
