# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""causal_conv1d_update torch emulator — gfx938 stopgap.

Third broken triton kernel found on DTK/gfx938 (after chunk_kda prefill):
``_causal_conv1d_update_kernel`` returns correlation-noise vs a python
reference (cos 0.26, weight-flip variant 0.50 — neither matches). It runs
on EVERY KDA layer at EVERY decode step (42 layers x steps), so it is the
decode-side word-soup driver once the prefill chunk_kda is emulated away.

Semantic (from the kernel source, W=4 non-spec arm): out = silu(bias +
s0*w0 + s1*w1 + s2*w2 + x*w3); conv_state slides left by 1 and appends x.

This emulator implements exactly that for the flags the KDA path uses
(2-D x single-token, conv_state_indices continuous batching, silu). The
torch math is exact; cost is trivial at (tokens, dim) decode sizes.
Selected via HCU_CONV1D_IMPL=emu (hip default; =fork routes to the vendored
fixed kernel — see fla_hcu_router).
"""

from __future__ import annotations

import torch


def causal_conv1d_update_emu(
    x: torch.Tensor,            # [batch, dim] (2-D) or [batch, dim, seqlen]
    conv_state: torch.Tensor,   # [cache_lines, dim, state_len], feat-major
    weight: torch.Tensor,       # [dim, width]
    bias: torch.Tensor | None,
    activation=None,
    conv_state_indices: torch.Tensor | None = None,
    **_ignored,
):
    squeeze = x.dim() == 2
    x3 = x.unsqueeze(-1) if squeeze else x
    batch, dim, seqlen = x3.shape
    _, width = weight.shape
    wf = weight.float()
    lines = conv_state.shape[0]
    state_len = conv_state.shape[-1]

    if conv_state_indices is None:
        slots = torch.arange(batch, device=x.device)
    else:
        slots = conv_state_indices.to(x.device).long()

    hist = torch.empty(batch, dim, seqlen + width - 1,
                       device=x.device, dtype=torch.float32)
    state_f = conv_state[slots].float()                # [batch, dim, state_len]
    hist[:, :, :width - 1] = state_f[:, :, -width + 1:]
    hist[:, :, width - 1:] = x3.float()
    # depthwise dot: out[b, d, t] = sum_k hist[b, d, t + k] * wf[d, k]
    out = torch.zeros(batch, dim, seqlen, device=x.device, dtype=torch.float32)
    for kk in range(width):
        out += hist[:, :, kk:kk + seqlen] * wf[:, kk].unsqueeze(0).unsqueeze(-1)
    if bias is not None:
        out += bias.float().view(1, -1, 1)
    if activation in ("silu", "swish", True):
        out = out * torch.sigmoid(out)

    # slide window: drop leftmost seqlen, append x
    new_state = torch.cat(
        [state_f[:, :, seqlen:], x3.float()], dim=-1
    ).to(conv_state.dtype)
    conv_state[slots] = new_state

    out = out.to(x3.dtype)
    return out.squeeze(-1) if squeeze else out


def install_patch():
    from sglang.kernels.ops.mamba import causal_conv1d_triton as m

    if getattr(m, "_hcu_convupd_emu", False):
        return
    _orig = m.causal_conv1d_update

    def wrapped(*args, **kwargs):
        # only the simple decode arm; spec-decode flag sets fall through
        if (
            kwargs.get("num_accept_tokens") is None
            and kwargs.get("intermediate_conv_window") is None
            and kwargs.get("metadata") is None
        ):
            return causal_conv1d_update_emu(
                args[0], args[1], args[2],
                args[3] if len(args) > 3 else kwargs.get("bias"),
                args[4] if len(args) > 4 else kwargs.get("activation", None),
                conv_state_indices=(
                    args[6] if len(args) > 6 and args[6] is not None
                    else kwargs.get("conv_state_indices")
                ),
            )
        return _orig(*args, **kwargs)

    m.causal_conv1d_update = wrapped
    m._hcu_convupd_emu = True
    m._hcu_convupd_orig = _orig
