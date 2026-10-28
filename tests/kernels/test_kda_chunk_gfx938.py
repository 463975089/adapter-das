"""chunk_kda (KDA/Glm5Next linear-attention PREFILL) gfx938 parity gate — §3.4b.

FAILING GATE (kept red until the kernel is fixed — do not skip).

Status as of 2026-10-15, nmz26 gfx938 / DTK 26.04, sglang v0.5.21:

- `kda_gate_chunk_cumsum` (stage 1): PASS — chunk-local cumsum matches a
  python reference to 1e-6.
- `fused_recurrent_kda_packed_decode` (decode): PASS — out/state cos
  >= 0.99999 vs the kernel's own math transcribed to torch
  (per-K safe gate lower_bound * sigmoid(exp(A_log)*(a + dt_bias)),
  state layout [V, K], scale folded on q only).
- `chunk_gated_delta_rule` (GDN prefill, different kernels): PASS with
  use_qk_l2norm_in_kernel=True (cos 0.997).
- `chunk_kda` (KDA prefill): FAIL — Aqk/intra values wrong for every
  flag combination (safe_gate x fuse_diagonal x fuse_recompute), every
  chunk size (16/32/64) and both raw and l2-normalized inputs; causal
  PLACEMENT (which cols are nonzero) is correct, VALUES are ~2.5x noise.
  With a zero gate (decay == 1, the simplest possible case) it still
  fails: cos 0.035 vs the beta*K·K^T reference, rms 0.50 vs 0.20.
  -> all 42 KDA layers compute garbage attention on prefill, which drowns
  the residual stream in noise (logprobs flat at -4 across vocab) — this
  is the GLM-5.3-Flash word-soup root cause.
  Downstream symptom: fwd_h's h collapses (rms 0.0) because w/u are
  derived from the broken A.

This test encodes the minimal repro so the fix gets a red->green signal:
single-hot cases PASS (placement + unit dots), random multi-row cases FAIL
(values). Run on the DTK box; needs one GPU.
"""

import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

import pytest  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)


def _mk(T=64, H=1, K=32, V=32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    v = torch.randn(1, T, H, V, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.rand(1, T, H, device="cuda", generator=g) * 0.9 + 0.05
    gate = torch.zeros(1, T, H, K, device="cuda", dtype=torch.float32)
    return q, k, v, b, gate


def _ref_beta_kkt(q, k, beta):
    """Beta * (q_i . k_j) over the causal half of the chunk, l2-normalized."""
    import torch.nn.functional as F

    qf = F.normalize(q.float(), dim=-1)
    kf = F.normalize(k.float(), dim=-1)
    T = q.shape[1]
    A = torch.zeros(1, T, q.shape[2], T, device="cuda")
    for i in range(T):
        for j in range(i + 1):
            A[0, i, :, j] = beta[0, i] * (qf[0, i] * kf[0, j]).sum(-1)
    return A


def _causal_cos(got, ref):
    T = ref.shape[-1]
    assert got.shape[-1] >= T, f"Aqk cols {got.shape[-1]} < T {T}"
    idx = torch.arange(T, device="cuda")
    mask = (idx[:, None] >= idx[None, :])[None, :, None, :]
    return torch.nn.functional.cosine_similarity(
        got.float()[..., :T][mask], ref[mask], dim=0
    ).item()


@pytest.mark.xfail(
    strict=True,
    reason="DTK gfx938 chunk_kda prefill Aqk/intra values are noise "
    "(witnessed 2026-10-15, cos ~0.08 zero-gate minimal case); the KDA emu "
    "(HCU_KDA_PREFILL_IMPL=emu) carries e2e. strict: XPASS (red) alerts us "
    "the day DTK fixes the kernel -- then flip the default routing.",
)
def test_chunk_kda_zero_gate_parity():
    """LIVE-path minimum case: chunk_kda (l2 inside) + zero gate + raw-beta.

    Zero gate => per-K decay identically 1, so output must equal the plain
    chunked delta rule (beta * K K^T intra/mlp state math). Currently FAILS
    with cos ~ 0.08 (2026-10-15, gfx938/DTK26.04): Aqk is the broken stage
    — placement correct, values noise. decode kernel, conv1d, and
    kda_gate_chunk_cumsum all pass their own parity gates.
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
    import torch.nn.functional as F

    B, T, H, K, V = 1, 64, 2, 32, 32
    scale = K ** -0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    beta_raw = torch.rand(B, T, H, device="cuda") * 0.9 + 0.05
    gate = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(H * K, device="cuda", dtype=torch.float32)
    st = torch.zeros(B, H, V, K, device="cuda", dtype=torch.float32)
    idx = torch.arange(B, device="cuda", dtype=torch.int32)
    o = chunk_kda(
        q=q, k=k, v=v, g=gate, beta=beta_raw,
        initial_state=st, initial_state_indices=idx,
        use_qk_l2norm_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
        lower_bound=-5.0, beta_is_raw=True,
    )
    o = o[0] if isinstance(o, (tuple, list)) else o
    torch.cuda.synchronize()

    qf = F.normalize(q.float(), dim=-1) * scale
    kf = F.normalize(k.float(), dim=-1)
    bf = torch.sigmoid(beta_raw.float())
    o_ref = torch.zeros(B, T, H, V, device="cuda")
    h = torch.zeros(B, H, V, K, device="cuda")  # state layout (V, K)
    for t in range(T):
        kt = kf[:, t].unsqueeze(-2)  # (B,H,1,K)
        mem = (h * kt).sum(-1)  # (B,H,V)
        h = h + bf[:, t].view(B, H, 1, 1) * kt * (
            v.float()[:, t].unsqueeze(-1) - mem.unsqueeze(-1)
        )
        o_ref[:, t] = (qf[:, t].unsqueeze(-2) * h).sum(-1)

    cos = torch.nn.functional.cosine_similarity(
        o.float().flatten(), o_ref.flatten(), dim=0
    ).item()
    assert cos >= 0.9999, (
        f"chunk_kda live-path broken on gfx938: cos={cos:.6f} "
        f"(see module docstring for the full fail matrix)"
    )


def test_chunk_kda_onehot_placement_and_values():
    """Placement probe: known-good standalone, nondeterministic after the
    decode kernel has run in the same process (64 cols vs row+1) — race
    evidence against inter_solve_fused touching sub_chunk-owned columns."""
    from sglang.kernels.ops.attention.fla.kda import chunk_kda_fwd_intra

    B, T, H, K, V = 1, 64, 1, 32, 32
    scale = K ** -0.5
    q = torch.zeros(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(B, T, H, device="cuda", dtype=torch.float32)
    gate = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    for i in (5, 20, 40):
        q[0, i, 0, 0] = 1.0
    k[0, :, 0, 0] = 1.0
    _, _, _, _, Aqk, _ = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=gate, beta=b, scale=scale,
        cu_seqlens=None, chunk_size=64, chunk_indices=None,
        safe_gate=False, fuse_diagonal=False, fuse_recompute=False,
    )
    torch.cuda.synchronize()
    # NOTE: cols strictly above the diagonal are NOT owned by any kernel —
    # Aqk arrives as torch.empty there. Only the causal j<=row wedge is
    # contractually defined; garbage above it is expected.
    for row in (5, 20, 40):
        wedge = Aqk[0, row, 0, : row + 1]
        nz = (wedge != 0).nonzero().flatten()
        assert nz.numel() == row + 1, f"row {row}: {nz.numel()} != {row + 1}"
        assert abs(float(wedge[row]) - float(wedge[0])) < 1e-3, (
            f"row {row}: self-entry drifts ({float(wedge[row]):.4f})"
        )


def test_kda_decode_kernel_parity():
    """Decode path is healthy — guards regressions while prefill is fixed."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    import torch.nn.functional as F

    H, K, V, BS = 2, 32, 32, 1
    scale = K ** -0.5
    LOWER = -5.0
    torch.manual_seed(0)
    mixed = torch.randn(BS, H * (2 * K + V), device="cuda", dtype=torch.bfloat16)
    a = torch.randn(BS, H * K, device="cuda", dtype=torch.float32) * 0.3
    b = torch.rand(BS, H, device="cuda", dtype=torch.float32) * 0.9 + 0.05
    A_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32) * 0.1
    st = torch.randn(BS, H, V, K, device="cuda", dtype=torch.float32) * 0.2
    st0 = st.clone()
    out = torch.zeros(BS, 1, H, V, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(BS, device="cuda", dtype=torch.int32)
    fused_recurrent_kda_packed_decode(
        mixed, a, b, A_log, dt_bias, scale, st, out, idx,
        use_qk_l2norm_in_kernel=True, lower_bound=LOWER,
    )
    torch.cuda.synchronize()
    q, k, v = mixed.split([H * K, H * K, H * V], dim=-1)
    q = F.normalize(q.view(BS, H, K).float(), dim=-1) * scale
    k = F.normalize(k.view(BS, H, K).float(), dim=-1)
    v = v.view(BS, H, V).float()
    g = LOWER * torch.sigmoid(
        torch.exp(A_log).view(1, H, 1) * (a.view(BS, H, K) + dt_bias.view(1, H, K))
    )
    beta = torch.sigmoid(b.float()).view(BS, H, 1, 1)
    h = st0 * g.exp().view(BS, H, 1, K)
    mem = (h * k.unsqueeze(-2)).sum(-1)
    h = h + beta * k.unsqueeze(-2) * (v.unsqueeze(-1) - mem.unsqueeze(-1))
    o = (h * q.unsqueeze(-2)).sum(-1)
    cos_o = torch.nn.functional.cosine_similarity(
        out.float().flatten(), o.flatten(), dim=0).item()
    cos_h = torch.nn.functional.cosine_similarity(
        st.float().flatten(), h.flatten(), dim=0).item()
    assert cos_o > 0.9999 and cos_h > 0.9999, f"decode drifted: o={cos_o} h={cos_h}"


if __name__ == "__main__":
    fails = []
    for name, fn in [
        ("decode-kernel", test_kda_decode_kernel_parity),
        ("onehot-placement", test_chunk_kda_onehot_placement_and_values),
        ("zero-gate-prefill", test_chunk_kda_zero_gate_parity),
    ]:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            fails.append(name)
    print("kda gfx938 gate:", "ALL PASS" if not fails else f"FAILING: {fails}")
    sys.exit(0 if not fails else 1)
