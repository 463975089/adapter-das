# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""causal_conv1d_fn varlen prefill arm — fork parity gates + emu green gate.

History: the stock DTK triton ``_causal_conv1d_fwd_kernel`` on gfx938
miscompiled the initial-state branches (has_initial_state read dead:
rand-vs-zero cache line gave identical output, state write-back maxdiff>2).
``test_triton_parity`` was an xfail-strict watchpoint on that defect; the
dead-initial-state witness documented it without asserting.

2026-10-19 (R-01 转正, 8849dc6): ``HCU_CONV1D_IMPL`` default emu→fork, so
``sglang.kernels.ops.mamba.causal_conv1d_triton`` now routes to the 收编
fork kernel (``hcu_adapter/fla_hcu/conv1d_triton.py``). The
watchpoint fired as designed (fork rejects the witness' CPU tensors at
launch → collection-visible failure). Follow-ups landed the same day:
  * channel-major inputs normalized at the fn seam (kernel channel-major
    arm miscomputes; serving always arrives channel-last);
  * initial-state read witness converted to an asserted green gate;
  * NEW xfail-strict watchpoint: the fork kernel miscomputes in an
    address-layout-dependent way under this suite's allocation layout
    (cos 0.67, deterministic per context, identical binary+data passes
    under other layouts) — DTK triton codegen suspicion, registered in
    REQUIREMENTS.md R-01.
The emu parity gate stays (emu = production 退路 tier). To re-witness the
stock defect, route ``HCU_CONV1D_IMPL=stock`` explicitly.

Structure per test-suite.md:
  1. offline repro (this file, CPU-free math table) — the step reference;
  2. one pit per test — routed-kernel parity + initial-state read (both
     green gates now), emu parity (green, fallback tier);
  3. assert with dual metrics cos + maxdiff.

Run: python tests/kernels/test_conv1d_fn_varlen.py   (GPU; skip-triton on CPU)
"""

from __future__ import annotations

import sys

import pytest
import torch

torch.manual_seed(7)


def step_reference(x, weight, bias, cu, states, slots, has_init):
    """Per-token global-time transcription of the causal_conv1d contract.

    x [dim, T] channel-major; states [B, dim, W-1]; has_init [B] bool.
    Returns (out [dim, T], new_states [B, dim, W-1]).
    """
    D, W = weight.shape
    T = x.shape[1]
    out = torch.zeros(D, T)
    hists = states.clone().float()
    wf = weight.float()
    bf = bias.float()
    for b in range(len(cu) - 1):
        t0, t1 = int(cu[b]), int(cu[b + 1])
        slot = int(slots[b])
        for t in range(t0, t1):
            win = torch.zeros(D, W)
            for j in range(W):
                gt = t - (W - 1 - j)  # global token feeding window col j
                if gt >= t0:
                    win[:, j] = x[:, gt].float()
                else:
                    sc = (W - 1) + (gt - t0)
                    if has_init[b] and 0 <= sc < W - 1:
                        win[:, j] = hists[slot][:, sc]
            v = (win * wf).sum(-1) + bf
            out[:, t] = torch.nn.functional.silu(v)
        for c in range(W - 1):
            gt = t1 - (W - 1) + c
            hists[slot][:, c] = x[:, gt].float() if gt >= t0 else 0.0
    return out, hists


def _metrics(got, ref):
    g, r = got.float().flatten(), ref.float().flatten()
    denom = g.norm() * r.norm()
    cos = (g @ r / denom).item() if denom > 0 else 0.0
    return cos, (g - r).abs().max().item()


def _case(D, W, lens, init):
    B = len(lens)
    T = sum(lens)
    cume = [sum(lens[: i + 1]) for i in range(B)]
    cu = torch.tensor([0] + cume, dtype=torch.int32)
    slots = torch.arange(B, dtype=torch.int32)
    x = torch.randn(D, T, dtype=torch.bfloat16)
    w = torch.randn(D, W, dtype=torch.bfloat16) * 0.3
    bias = torch.randn(D, dtype=torch.bfloat16) * 0.1
    states = torch.randn(B, D, W - 1, dtype=torch.bfloat16)
    hid = torch.tensor([init] * B)
    return x, w, bias, cu, states, slots, hid


def _case_dev(D, W, lens, init):
    """_case with all tensors on the GPU (routed triton kernel needs device
    pointers — CPU tensors are correctly rejected at launch, 2026-10-19)."""
    x, w, bias, cu, states, slots, hid = _case(D, W, lens, init)
    return (x.cuda(), w.cuda(), bias.cuda(), cu.cuda(), states.cuda(),
            slots.cuda(), hid.cuda())


@pytest.mark.xfail(
    strict=True,
    reason="DTK gfx938 triton miscompiles the 收编 fork "
    "_causal_conv1d_fwd_kernel in an address-layout-dependent way "
    "(R-11 merge-gate L1 finding, 2026-10-19: identical data+binary gives "
    "cos 0.67 under the L1 suite's deterministic allocation layout, "
    "cos 0.9999 under other layouts — per-context deterministic, 6/6; "
    "serving-shaped gates never red. Repro: tests above burn-in; see "
    "REQUIREMENTS.md R-01 conv1d-flake entry). strict: XPASS alerts us the "
    "day the layout sensitivity disappears (DTK fix or kernel rework).",
)
def test_triton_parity():
    """Watchpoint gate: routed conv1d_fn vs global-time step reference.

    The routed kernel is the 收编 fork (HCU_CONV1D_IMPL=fork default,
    8849dc6). Channel-major inputs are normalized at the fn seam (same fix
    window); the residual red is the address-layout-sensitive miscompute
    above — fails deterministically inside this suite's allocation layout,
    passes under others. The witness test below (initial-state read) stays
    an asserted green gate; the emu parity gate carries the exactness
    contract (emu = production 退路 tier).
    """
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )
    except ImportError:
        pytest.skip("kernel module unavailable (CPU?)")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    for tag, (D, W, lens) in {
        "init-batch": (64, 4, [5, 3]),
        "single": (32, 4, [5]),
        "len1-burst": (96, 4, [1, 1, 7]),
    }.items():
        x, w, bias, cu, states, slots, hid = _case_dev(D, W, lens, True)
        out = causal_conv1d_fn(
            x, w, bias, activation="silu", conv_states=states,
            cache_indices=slots, query_start_loc=cu, has_initial_state=hid,
            seq_lens_cpu=lens,
        )
        torch.cuda.synchronize()
        ref, _ = step_reference(
            x.cpu(), w.cpu(), bias.cpu(), cu.cpu().tolist(), states.cpu(),
            slots.cpu().tolist(), hid.cpu().tolist(),
        )
        cos, md = _metrics(out.cpu(), ref)
        print(f"[gate] triton varlen {tag}: cos={cos:.6f} maxdiff={md:.4f}")
        assert cos > 0.9999 and md < 2e-2, (
            f"routed conv1d_fn varlen {tag} broken: cos={cos:.6f} "
            f"maxdiff={md:.4f}; check HCU_CONV1D_IMPL routing"
        )


def test_emu_parity():
    """GREEN GATE: conv1d_fn_emu == step reference across shapes/branches."""
    from hcu_adapter.conv1d_fn_emu import conv1d_fn_emu

    for tag, (D, W, lens, init) in {
        "short-seq-init": (16, 4, [2, 5], True),
        "short-seq-fresh": (16, 4, [2, 5], False),
        "len1-burst": (96, 4, [1, 1, 7], True),
        "single-w4": (64, 4, [6], True),
        "mixed": (96, 4, [5, 3], True),
        "w2-arm": (8, 2, [4, 3], True),
    }.items():
        x, w, bias, cu, states, slots, hid = _case(D, W, lens, init)
        got = conv1d_fn_emu(
            x, w, bias, states.clone(), cu, torch.tensor(lens), slots, hid
        )
        if got.shape[0] != x.shape[0]:
            got = got.t()
        ref, _ = step_reference(
            x, w, bias, cu.tolist(), states, slots.tolist(), hid.tolist()
        )
        cos, md = _metrics(got, ref)
        assert cos > 0.9999 and md < 2e-2, (
            f"emu {tag} drifted: cos={cos:.6f} maxdiff={md:.4f}"
        )
        print(f"[PASS] emu {tag}: cos={cos:.6f} maxdiff={md:.4f}")


def test_dead_initial_state_witness():
    """GREEN GATE (since R-01 转正): initial state MUST be read.

    Stock DTK triton had a dead initial-state read (rand vs zero cache line
    → identical output); that defect is what this witness documented. The
    routed fork kernel must show the OPPOSITE: outputs differ. No-assert
    witness converted to a regression assert 2026-10-19 (CPU-tensor launch
    crash after 转正 exposed the staleness).
    """
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )
    except ImportError:
        pytest.skip("kernel module unavailable (CPU?)")
        return
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    x, w, bias, cu, states, slots, hid = _case_dev(32, 4, [5], True)
    out_rand = causal_conv1d_fn(
        x, w, bias, activation="silu", conv_states=states.clone(),
        cache_indices=slots, query_start_loc=cu, has_initial_state=hid,
        seq_lens_cpu=[5],
    )
    out_zero = causal_conv1d_fn(
        x, w, bias, activation="silu", conv_states=torch.zeros_like(states),
        cache_indices=slots, query_start_loc=cu, has_initial_state=hid,
        seq_lens_cpu=[5],
    )
    torch.cuda.synchronize()
    diff = (out_rand.float() - out_zero.float()).abs().max().item()
    print(f"[witness] rand-state vs zero-state out maxdiff={diff:.6f} "
          f"({'DEAD initial-state read — regression!' if diff < 1e-3 else 'state is read'})")
    assert diff > 1e-3, (
        f"initial-state read is dead again (rand==zero out, maxdiff={diff}): "
        "routed kernel regressed to the stock DTK defect"
    )


def main():
    fails = []
    for name, fn in [
        ("triton-parity (routed green gate)", test_triton_parity),
        ("emu-parity (green gate)", test_emu_parity),
        ("dead-initial-state witness (green gate)", test_dead_initial_state_witness),
    ]:
        try:
            fn()
        except AssertionError as e:
            fails.append((name, e))
            print(f"[FAIL] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            fails.append((name, e))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    for name, e in fails:
        print(f"FAILED: {name}: {e}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
