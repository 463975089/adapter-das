"""Flash-attention vendor stress across the scenario matrix (HCU adapter).

FA kernels break on FLEXIBLE shapes, not on the vendor's own unit-test
shapes: odd seqlens, qo != kv, varlen packing, head_dim 64 vs 128, causal
vs not, and CUDA-graph replay with fresh inputs (the same detector that
caught the aiter AR replay corruption). This file drives every FA backend
that is importable on the host through the SAME matrix against an fp32
torch reference:

  backends probed (first wins; add ours when sgl_kernel ships an FA port):
    aiter.flash_attn_varlen_func   (DTK vendor wheel, if present)

Run: pytest tests/test_flash_attn_gpu.py -v   (1+ GPU).
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

class AiterFA:
    name = "aiter.flash_attn_varlen_func"
    varlen = True

    def __init__(self):
        import aiter

        self.fn = aiter.flash_attn_varlen_func

    def __call__(self, q, k, v, causal, scale):
        # q [Tq,H,D] k/v [Tk,H,D] flattened varlen, single sequence
        tq, h, d = q.shape
        tk = k.shape[0]
        cq = torch.tensor([0, tq], dtype=torch.int32, device=q.device)
        ck = torch.tensor([0, tk], dtype=torch.int32, device=q.device)
        out, _ = self.fn(
            q, k, v, cq, ck, tq, tk,
            dropout_p=0.0, softmax_scale=scale, causal=causal,
        )
        return out


def load_backends():
    out = []
    try:
        out.append(AiterFA())
    except Exception as e:
        print(f"[fa-stress] aiter FA unavailable: {e!r}")
    return out


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
        pytest.skip("no FA backend importable on this host")
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
        pytest.skip("no FA backend importable on this host")
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
