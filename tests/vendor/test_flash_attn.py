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
        # Shape-only metadata must not enqueue pageable CPU copies during
        # capture. Build the same CU values on device, as the dense bridge does.
        cq = torch.arange(2, dtype=torch.int32, device=q.device) * tq
        ck = torch.arange(2, dtype=torch.int32, device=q.device) * tk
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
    o = b(*(t[0].transpose(0, 1).contiguous() for t in (q, k, v)),
          True, scale).transpose(0, 1)[None].float()
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
    # Bottom-right causal attention permits fully masked rows when Tq>Tk.
    # FA defines their output as zero. Avoid NaN in the independent oracle.
    valid = torch.isfinite(att).any(dim=-1, keepdim=True)
    probabilities = F.softmax(torch.where(valid, att, torch.zeros_like(att)), dim=-1)
    return (probabilities * valid) @ v


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
                q[0].transpose(0, 1).contiguous(),
                k[0].transpose(0, 1).contiguous(),
                v[0].transpose(0, 1).contiguous(),
                causal, scale,
            )
        except NotImplementedError as e:
            pytest.skip(f"{b.name} rejects this shape: {e!r}")
        check(
            f"{b.name} qo={qo} kv={kv} hd={hd} "
            f"{str(dtype).split('.')[-1]} causal={causal}",
            out.transpose(0, 1)[None], ref,
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
                    nq.transpose(0, 1)[None].float(),
                    nk.transpose(0, 1)[None].float(),
                    nv.transpose(0, 1)[None].float(), True, scale,
                )
            check(f"{b.name} graph-replay[{trial}]", out.transpose(0, 1)[None], ref)


@pytest.fixture
def fallback_host(monkeypatch):
    import sys
    import types
    from hcu_adapter.infra import hcu_env, registry

    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(hcu_env, "resolve", lambda name: ("vendor", "explicit"))
    monkeypatch.setattr(hcu_env, "arch_name", lambda: "gfx936")
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    modules = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.arg_groups",
                 "sglang.srt.arg_groups.overrides", "sglang.srt.arg_groups.attention_hook"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], attr, module)
    overrides = modules["sglang.srt.arg_groups.overrides"]
    consumer = modules["sglang.srt.arg_groups.attention_hook"]

    def original(view):
        return {"fallback": (view.attention_backend, view.kv_cache_dtype)}

    overrides._attention_backend_fa3_fp8_fallback = original
    consumer._attention_backend_fa3_fp8_fallback = original
    overrides.POST_PROCESS_PASSES = [original, object(), original]
    return overrides, consumer, original


def test_e5m2_guard_updates_consumers_and_preserves_fallback(fallback_host):
    from types import SimpleNamespace
    from hcu_adapter.bridges import sglang_fa_vendor as fa_vendor

    overrides, consumer, original = fallback_host
    fa_vendor._install_e5m2_backend_guard()
    installed = overrides._attention_backend_fa3_fp8_fallback
    assert installed is consumer._attention_backend_fa3_fp8_fallback
    assert overrides.POST_PROCESS_PASSES[0] is installed
    assert overrides.POST_PROCESS_PASSES[2] is installed
    assert installed(SimpleNamespace(attention_backend="fa3", kv_cache_dtype="fp8_e5m2")) == {}
    for backend, dtype in (("fa3", "fp8_e4m3"), ("triton", "fp8_e5m2")):
        view = SimpleNamespace(attention_backend=backend, kv_cache_dtype=dtype)
        assert installed(view) == original(view)
    fa_vendor._install_e5m2_backend_guard()
    assert overrides._attention_backend_fa3_fp8_fallback is installed


@pytest.mark.parametrize("mode,arch,hip", [
    ("auto", "gfx936", "test"), ("triton", "gfx936", "test"),
    ("emu", "gfx936", "test"), ("vendor", "gfx942", "test"),
    ("vendor", "gfx936", None),
])
def test_e5m2_guard_disabled_keeps_all_bindings(fallback_host, monkeypatch, mode, arch, hip):
    from hcu_adapter.bridges import sglang_fa_vendor as fa_vendor
    from hcu_adapter.infra import hcu_env, registry

    overrides, consumer, original = fallback_host
    monkeypatch.setattr(hcu_env, "resolve", lambda name: (mode, "explicit"))
    monkeypatch.setattr(hcu_env, "arch_name", lambda: arch)
    monkeypatch.setattr(torch.version, "hip", hip)
    fa_vendor._install_e5m2_backend_guard()
    assert overrides._attention_backend_fa3_fp8_fallback is original
    assert consumer._attention_backend_fa3_fp8_fallback is original
    assert overrides.POST_PROCESS_PASSES[0] is original
    assert registry.get("flash_attention") is None


def test_e5m2_guard_registration_failure_is_atomic(fallback_host, monkeypatch):
    from hcu_adapter.bridges import sglang_fa_vendor as fa_vendor
    from hcu_adapter.infra import registry

    overrides, consumer, original = fallback_host
    register = registry.register
    count = 0

    def reject_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("registration refused")
        return register(*args, **kwargs)

    monkeypatch.setattr(registry, "register", reject_second)
    with pytest.raises(ValueError, match="registration refused"):
        fa_vendor._install_e5m2_backend_guard()
    assert overrides._attention_backend_fa3_fp8_fallback is original
    assert consumer._attention_backend_fa3_fp8_fallback is original
    assert overrides.POST_PROCESS_PASSES[0] is original
    assert registry.get("flash_attention") is None
