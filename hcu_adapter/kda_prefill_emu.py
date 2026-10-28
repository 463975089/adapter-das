"""KDA prefill (chunk_kda) CPU/GPU-torch emulator — gfx938 stopgap.

The triton ``chunk_kda`` pipeline computes a wrong Aqk on gfx938 (noise
level cos vs a step reference even with zero gate; see
``tests/kernels/test_kda_chunk_gfx938.py`` — that gate stays RED for the
triton kernels and GREEN for this emulator). All 42 KDA layers of
GLM-5.3-Flash route through it on prefill, so the whole model degrades to
noise logits. This module replaces only ``TritonKDAKernel.extend`` with a
sequential (per-token) torch emulation of the exact same math the decode
kernel implements and which passes its parity gate:

    state h [H, V, K] (per K-col gate):
        h = h * exp(g_t)                     # g_t per-K safe gate
        mem = (h * k_t).sum(K)
        delta = v_t - mem
        h += beta_t * k_t (outer) * delta
        o_t = (q_t scale) (row) . h

conventions (validated against the triton decode):
- q/k l2-normalized per head, q additionally scaled by K^-0.5;
- gate g_raw [B, T, H, K] raw, activated in-kernel as
  ``lower_bound * sigmoid(exp(A_log_h) * (g_raw + dt_bias))`` when
  lower_bound set, else ``-exp(A_log_h) * softplus(g_raw + dt_bias)``;
- beta raw -> sigmoid;
- state layout [num_slots, H, V, K], slot picked via initial_state_indices;
- final state written back to the same slot.

Varlen via cu_seqlens when provided (extend batches), else dense [B, T].
Selected via ``HCU_KDA_PREFILL_IMPL=emu`` (hip default; the =fork option
and its serving-gate caveat live in fla_hcu_router). Not for production
throughput — sequential in
T; meant to unblock functional bring-up.
"""

from __future__ import annotations

import torch


def _activate(g_raw, A_log, dt_bias, seq_lens, lower_bound, max_T):
    """Per-K gate activation, one step of the recurrence decay."""
    H = A_log.numel()
    if g_raw.dim() == 3:  # [T, H*K] varlen flattened
        K = g_raw.shape[-1] // H
        g_raw = g_raw.view(1, g_raw.shape[0], H, K)
    else:
        K = g_raw.shape[-1]
    x = g_raw + dt_bias.view(1, 1, H, K).to(g_raw.dtype)
    if lower_bound is not None:
        g = lower_bound * torch.sigmoid(
            torch.exp(A_log).view(1, 1, -1, 1) * x.float()
        )
    else:
        g = -torch.exp(A_log).view(1, 1, -1, 1) * torch.nn.functional.softplus(
            x.float()
        )
    return g


def kda_extend_emu(
    q: torch.Tensor,      # [B, T, HV, K] or [1, T_varlen, HV, K]
    k: torch.Tensor,
    v: torch.Tensor,      # [B, T, HV, V]
    g: torch.Tensor,      # raw gate [B, T, HV, K]
    beta: torch.Tensor,   # raw beta [B, T, HV]
    *,
    ssm_states: torch.Tensor,          # [num_slots, HV, V, K]
    cache_indices: torch.Tensor,       # [B]
    query_start_loc: torch.Tensor,     # [B+1] int32
    A_log: torch.Tensor,               # [HV]
    dt_bias: torch.Tensor,             # [HV*K]
    lower_bound: float | None,
    scale: float,
    beta_is_raw: bool = True,
    **_ignored,
):
    B = query_start_loc.numel() - 1
    device = q.device
    HV, V, K = ssm_states.shape[-3:]
    T_total = q.shape[1]

    g_act = _activate(g, A_log, dt_bias, None, lower_bound, T_total)
    beta = beta.float().sigmoid() if beta_is_raw else beta.float()

    q_n = torch.nn.functional.normalize(q.float(), dim=-1) * scale
    k_n = torch.nn.functional.normalize(k.float(), dim=-1)

    o = torch.empty_like(q_n[..., :V])
    for b in range(B):
        t0 = int(query_start_loc[b])
        t1 = int(query_start_loc[b + 1])
        slot = int(cache_indices[b])
        h = ssm_states[slot].float().clone()  # [HV, V, K]
        for t in range(t0, t1):
            dk = g_act[0, t].exp().unsqueeze(-2)          # [HV, 1, K]
            h = h * dk
            kt = k_n[0, t].unsqueeze(-2)                  # [HV, 1, K]
            mem = (h * kt).sum(-1, keepdim=True)          # [HV, V, 1]
            delta = v[0, t].float().unsqueeze(-1) - mem   # [HV, V, 1]
            h = h + beta[0, t].view(HV, 1, 1) * kt * delta
            o[0, t] = (h * q_n[0, t].unsqueeze(-2)).sum(-1)
        ssm_states[slot] = h.to(ssm_states.dtype)
    return o.to(torch.bfloat16), None


def install_patch():
    """Wrap TritonKDAKernel.extend with the emulator (env-gated at caller)."""
    from sglang.srt.layers.attention.linear.kernels.kda_triton import (
        TritonKDAKernel,
    )

    if getattr(TritonKDAKernel, "_hcu_emu_patched", False):
        return
    _orig = TritonKDAKernel.extend

    def extend(self, *args, **kwargs):
        # kda_backend extend() calls this positionally in places: accept both.
        if args:
            # (q, k, v, g, beta) then keyword-only configuration
            q, k, v, g, beta = args[:5]
            kwargs = {**kwargs}
            kwargs.setdefault("q", q); kwargs.setdefault("k", k)
            kwargs.setdefault("v", v); kwargs.setdefault("g", g)
            kwargs.setdefault("beta", beta)
        q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
        g, beta = kwargs["g"], kwargs["beta"]
        common = dict(
            ssm_states=kwargs["ssm_states"],
            cache_indices=kwargs["cache_indices"],
            query_start_loc=kwargs["query_start_loc"],
            A_log=kwargs.get("A_log"),
            dt_bias=kwargs.get("dt_bias"),
            lower_bound=kwargs.get("lower_bound"),
            scale=q.shape[-1] ** -0.5,
            beta_is_raw=kwargs.get("beta_is_raw", False),
        )
        if kwargs.get("return_intermediate_states"):
            # track path unpacks (o, h); emulate the h snapshot as the
            # final state broadcast to the [1] chunk dim.
            o, h_state = kda_extend_emu(q, k, v, g, beta, **common)
            if h_state is not None:
                return o, h_state.unsqueeze(0)
            return o, None
        # Plain path must return the TENSOR (the model feeds core_attn_out
        # straight into rms_norm_gated); the emu returns (o, None).
        o, _h = kda_extend_emu(q, k, v, g, beta, **common)
        return o

    TritonKDAKernel.extend = extend
    TritonKDAKernel._hcu_emu_patched = True
    TritonKDAKernel._hcu_emu_orig_extend = _orig
