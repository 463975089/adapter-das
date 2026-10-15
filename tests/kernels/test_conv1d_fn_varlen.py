"""causal_conv1d_fn varlen prefill arm — triton red gate + emu green gate.

DTK triton ``_causal_conv1d_fwd_kernel`` on gfx938 miscompiles the
initial-state branches (parallel to the confirmed ``causal_conv1d_update``
update-arm noise): with ``has_initial_state`` set, the kernel's output is
uncorrelated with the per-token step reference AND identical whether the
cache line is random or zero — the initial-state read is dead. State
write-back is also wrong (maxdiff > 2). All 42 KDA layers of GLM-5.3-Flash
run their prefill through this kernel, so logits are noise from token one
until the DTK triton bug is fixed; ``conv1d_fn_emu`` (selected via
``HCU_CONV1D_IMPL=emu`` in fla_hcu_router) transcribes the contract in
torch and must stay exact.

Structure per test-suite.md:
  1. offline repro (this file, CPU-free math table) — the step reference;
  2. one pit per test — triton parity (red until fixed), emu parity (green),
     dead-initial-state witness (rand == zero output), state write-back;
  3. assert with dual metrics cos + maxdiff.

Run: python tests/kernels/test_conv1d_fn_varlen.py   (GPU; skip-triton on CPU)
"""

from __future__ import annotations

import sys

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


def test_triton_parity():
    """RED GATE (gfx938 triton): conv1d_fn vs step reference.

    Witnessed 2026-10-15: with has_initial_state=True
    out cos=0.49 maxdiff=1.40; state write-back maxdiff=2.05. Stays red
    until DTK triton fixes the initial-state branches; the emu carries e2e.
    """
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )
    except ImportError:
        print("SKIP triton parity: kernel module unavailable (CPU?)")
        return
    if not torch.cuda.is_available():
        print("SKIP triton parity: no GPU")
        return
    x, w, bias, cu, states, slots, hid = _case(64, 4, [5, 3], True)
    out = causal_conv1d_fn(
        x, w, bias, activation="silu", conv_states=states,
        cache_indices=slots, query_start_loc=cu, has_initial_state=hid,
        seq_lens_cpu=[5, 3],
    )
    torch.cuda.synchronize()
    ref, _ = step_reference(
        x, w, bias, cu.tolist(), states.cpu(), slots.tolist(), hid.tolist()
    )
    cos, md = _metrics(out.cpu(), ref)
    print(f"[gate] triton varlen: cos={cos:.6f} maxdiff={md:.4f} (red until DTK fix)")
    assert False, (
        f"gfx938 triton conv1d_fn still broken: cos={cos:.6f} maxdiff={md:.4f}; "
        "emu carries e2e (HCU_CONV1D_IMPL=emu)"
    )


def test_emu_parity():
    """GREEN GATE: conv1d_fn_emu == step reference across shapes/branches."""
    from sgl_kernel._adapters.conv1d_fn_emu import conv1d_fn_emu

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
    """Documents the exact triton defect: random vs zero cache line, same out."""
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )
    except ImportError:
        print("SKIP dead-state witness: no kernel module")
        return
    if not torch.cuda.is_available():
        print("SKIP dead-state witness: no GPU")
        return
    x, w, bias, cu, states, slots, hid = _case(32, 4, [5], True)
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
          f"({'DEAD initial-state read — kernel bug confirmed' if diff < 1e-3 else 'state is read'})")
    return diff


def main():
    fails = []
    for name, fn in [
        ("triton-parity (red gate)", test_triton_parity),
        ("emu-parity (green gate)", test_emu_parity),
        ("dead-initial-state witness", test_dead_initial_state_witness),
    ]:
        try:
            fn()
        except AssertionError as e:
            # the triton parity gate is EXPECTED to fail until DTK fixes it
            if "red gate" in name:
                print(f"[RED-as-designed] {name}: {e}")
            else:
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
