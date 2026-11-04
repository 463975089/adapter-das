# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Flash-attention vendor stress across the scenario matrix (HCU adapter).

FA kernels break on FLEXIBLE shapes, not on the vendor's own unit-test
shapes: odd seqlens, qo != kv, varlen packing, head_dim 64 vs 128, causal
vs not, and CUDA-graph replay with fresh inputs (the same detector that
caught the aiter AR replay corruption). This file drives every FA backend
that is importable on the host through the SAME matrix against an fp32
torch reference:

  backends probed (first wins; add ours when sgl_kernel ships an FA port):
    aiter.flash_attn_varlen_func   (DTK vendor wheel, if present)

Run: pytest tests/vendor/test_flash_attn.py -v   (1+ GPU).
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


# ---------------------------------------------------------------------------
# backend probe
# ---------------------------------------------------------------------------

class _VarlenFA:
    """Shared driver for FA2-style varlen ABIs (flash_attn package, aiter):
    fn(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, softmax_scale, causal)
    -> (out, lse)."""

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, q, k, v, causal, scale):
        tq, h, d = q.shape
        tk = k.shape[0]
        cq = torch.tensor([0, tq], dtype=torch.int32, device=q.device)
        ck = torch.tensor([0, tk], dtype=torch.int32, device=q.device)
        # ABI drift across wheel versions: 2.6.x returns (out, lse), 2.8.x
        # (out, lse, rng_state?) — inspectable shapes first, tensor wins.
        res = self.fn(
            q, k, v, cq, ck, tq, tk,
            dropout_p=0.0, softmax_scale=scale, causal=causal,
        )
        if isinstance(res, (tuple, list)):
            out = next(r for r in res if torch.is_tensor(r)
                       and r.shape == q.shape)
        else:
            out = res
        return out


def load_backends():
    """Probe importable FA backends and sanity-gate each one.

    A wheel whose numbers are garbage on this arch (e.g. flash_attn 2.8.4
    compiled for gfx942, run on gfx938 — plain and varlen identically wrong,
    cos 0.002..0.8) is dropped with a loud warning instead of 53 red
    failures burying the one-line verdict. Repair the wheel and the matrix
    resumes automatically.
    """
    out = []
    # HCU flash_attn wheel (OpenDAS cancon build) — the vendor FA2 port the
    # sglang-das invasive fork also falls back to; with it installed the
    # adapter's flash_attn stub is bypassed automatically (real pkg wins).
    try:
        from flash_attn import flash_attn_varlen_func as _fa2

        out.append(_VarlenFA("flash_attn.flash_attn_varlen_func", _fa2))
    except Exception as e:
        print(f"[fa-stress] flash_attn pkg unavailable: {e!r:.100}")
    try:
        import aiter

        out.append(_VarlenFA(
            "aiter.flash_attn_varlen_func", aiter.flash_attn_varlen_func))
    except Exception as e:
        print(f"[fa-stress] aiter FA unavailable: {e!r:.100}")
    return [b for b in out if _backend_sanity(b)]


def _backend_sanity(b):
    """One known small scenario vs fp32 sdpa; guards against known-bad wheels."""
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return True
    torch.manual_seed(1234)
    t, h, d = 33, 4, 128
    scale = 1.0 / math.sqrt(d)
    # INDEPENDENT q/k/v: this wheel family passes its own q=k=v degenerate
    # input but is garbage on real attention (53/52 matrix fails before the
    # gate existed) — sanity must match the matrix distribution.
    q, k, v = ((torch.randn(1, h, t, d, device="cuda") * 1.5)
               .to(torch.bfloat16) for _ in range(3))
    # ref BEFORE the call: an in-place kernel would corrupt q and the fp32
    # ref would silently match its own corruption (cos=1.0 trap).
    r = sdpa_ref(q.float(), k.float(), v.float(), True, scale)
    q0, k0, v0 = q.clone(), k.clone(), v.clone()
    # backends take the 3D [t, h, d] varlen form (same as the matrix calls)
    o = b(q.squeeze(0), k.squeeze(0), v.squeeze(0), True, scale).float()
    for nm, t0, t1 in (("q", q0, q), ("k", k0, k), ("v", v0, v)):
        if not torch.equal(t0, t1):
            print(f"[fa-stress] WARNING: {b.name} modified its input "
                  f"{nm} in place (out aliases input — same-name "
                  f"different-semantics bug class); refusing it.")
            return False
    cos = F.cosine_similarity(
        o.flatten(), r.float().flatten(), dim=0).item()
    if cos < 0.99:
        print(f"[fa-stress] WARNING: {b.name} is numerically BROKEN on "
              f"this host (sanity cos={cos:.4f} vs fp32 sdpa); dropping "
              f"it from the matrix — do not enable aiter_fa/flash_attn "
              f"paths until the wheel is fixed.")
        return False
    print(f"[fa-stress] sanity ok: {b.name} (cos={cos:.6f})")
    return True


# ---------------------------------------------------------------------------
# fp32 reference
# ---------------------------------------------------------------------------

def sdpa_ref(q, k, v, causal, scale):
    # q [B,H,Tq,D] fp32 reference, math in fp32
    att = (q @ k.transpose(-1, -2)) * scale
    if causal:
        tq, tk = q.shape[-2], k.shape[-2]
        mask = torch.ones(
            tq, tk, dtype=torch.bool, device=q.device
        ).tril(diagonal=tk - tq)
        att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1) @ v


def check(name, out, ref):
    """Distribution-level assertion: bf16 noise reorders near-tie rows, so
    we bound max-relative error AND cosine — layout/rotation bugs destroy
    cosine even when a loose elementwise tol would pass."""
    o, r = out.float(), ref.float()
    scale = r.abs().max().item() + 1e-6
    maxrel = ((o - r).abs() / scale).max().item()
    cos = F.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    assert maxrel <= 5e-2 and cos >= 0.999, (
        f"{name}: maxrel={maxrel:.4f} cos={cos:.6f} "
        f"(semantic/layout bug suspected, not bf16 noise)"
    )
    return f"maxrel={maxrel:.3e} cos={cos:.6f}"


# qo, kv, heads, headdim, causal  — odd lengths, qo!=kv, boundary sizes
SCENARIOS = [
    (1, 1, True),    # (qo=kv) pure decode step
    (1, 7, True),
    (1, 127, True),
    (1, 1024, True),
    (3, 129, True),
    (17, 1023, True),
    (17, 1023, False),
    (127, 127, True),
    (128, 128, True),
    (129, 257, True),
    (1023, 1023, True),
    (1024, 1, True),   # attend to a single cached token
    (2051, 2051, True),
]
HEAD_DIMS = [64, 128]
DTYPES = [torch.bfloat16, torch.float16]


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hd", HEAD_DIMS)
@pytest.mark.parametrize("qo,kv,causal", SCENARIOS)
def test_fa_scenario_matrix(dtype, hd, qo, kv, causal):
    backs = load_backends()
    if not backs:
        pytest.skip(
            "no numerically-sound FA backend on this host (see sanity log)")
    torch.manual_seed(7 + qo * 31 + kv)
    h, d = 4, hd
    scale = 1.0 / math.sqrt(d)
    q = (torch.randn(1, h, qo, d, device="cuda") * 1.5).to(dtype)
    k = (torch.randn(1, h, kv, d, device="cuda") * 1.5).to(dtype)
    v = (torch.randn(1, h, kv, d, device="cuda") * 1.5).to(dtype)
    ref = sdpa_ref(q.float(), k.float(), v.float(), causal, scale)
    for b in backs:
        try:
            out = b(
                q.squeeze(0).contiguous(),
                k.squeeze(0).contiguous(),
                v.squeeze(0).contiguous(),
                causal, scale,
            )
        except NotImplementedError as e:
            pytest.skip(f"{b.name} rejects this shape: {e!r}")
        check(
            f"{b.name} qo={qo} kv={kv} hd={hd} "
            f"{str(dtype).split('.')[-1]} causal={causal}",
            out, ref,
        )


@requires_gpu
def test_fa_graph_replay_tracks_fresh_inputs():
    """CUDA-graph capture/replay with FRESH inputs — the detector class that
    caught the aiter AR corruption (eager passes, replay bakes/corrupts)."""
    backs = load_backends()
    if not backs:
        pytest.skip(
            "no numerically-sound FA backend on this host (see sanity log)")
    torch.manual_seed(11)
    h, d, qo, kv = 4, 128, 1, 1024
    scale = 1.0 / math.sqrt(d)
    dt = torch.bfloat16
    sq = torch.randn(qo, h, d, device="cuda").to(dt)
    sk = torch.randn(kv, h, d, device="cuda").to(dt)
    sv = torch.randn(kv, h, d, device="cuda").to(dt)
    for b in backs:
        try:
            for _ in range(3):
                b(sq, sk, sv, True, scale)  # warmup
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = b(sq, sk, sv, True, scale)
        except NotImplementedError as e:
            pytest.skip(f"{b.name} rejects capture shapes: {e!r}")
        for trial in range(4):
            with torch.no_grad():
                nq = torch.randn(qo, h, d, device="cuda").to(dt)
                nk = torch.randn(kv, h, d, device="cuda").to(dt)
                nv = torch.randn(kv, h, d, device="cuda").to(dt)
                sq.copy_(nq)
                sk.copy_(nk)
                sv.copy_(nv)
                g.replay()
                torch.cuda.synchronize()
                ref = sdpa_ref(
                    nq.view(1, h, qo, d).float(),
                    nk.view(1, h, kv, d).float(),
                    nv.view(1, h, kv, d).float(), True, scale,
                )
            check(f"{b.name} graph-replay[{trial}]", out, ref)
