# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""W4A16 MoE bridge, moe_c tier: DTK aiter moe_c w4a16 marlin vendor GEMMs
under the stock triton ``fused_experts`` face (box16 dsv41f-INT4, 2026-10-26).

Why
---
The production INT4 line (DeepSeek-V4.1-Flash-INT4, channel-wise
compressed-tensors, E=48/rank H=5120 I=2304 topk=6) rides the stock triton
``fused_moe_kernel_gptq_awq`` whose decode cost is expert-count-driven and
latency-bound: 5.44ms UP + 2.70ms DOWN per layer event-timed (trace shows
~7us under-reported, follower kernels absorb the bubble time -- driver
bubble-absorption fix, docs/cookbook.md §5). No triton config is sub-ms
(sweep floor 3.9ms). The DTK aiter wheel ships an unrelated vendor family
``aiter.moe_c_moe_gemm_marlin_w4a16`` (raw packed weights as uint32,
bf16 group-32 scales, marlin scale shuffle, zeros arg required-but-ignored
for the symmetric (q-8)*s convention) that this bridge drives directly:
align -> vendor GEMM1 -> exact dsv4 clamp-silu -> vendor GEMM2 -> sum.

box16 (gfx938, dtk26041 image wheel) probe ladder, all parity vs a torch
dequant reference with per-slot unique scales:
  * golden shape (DTK's own test E=384 H=7168 I=256 tk8): med rel 0.006;
  * dim morph: I=2304 alone breaks parity -> gemm2 K law;
  * gemm2 works iff (K/32) % 16 == 0 (K=256 special-case also ok): I in
    {256,512,1024,1536,2048} pass, {384,768,2304} fail;
  * fix: pad gemm2 K to a multiple of 512 with nibble 8 (dequant (8-8)*s
    = 0, mathematically exact) and zero-pad the activation columns;
    2304 -> 2560 (+11% gemm2 flops). gemm1 needs no pad (H=5120, 160%16=0,
    N=4608 %256=0 for the MODE-4 tile).
  * full pipeline med rel 0.0055 at M=2/6/16; per-layer full MoE 460/745/
    1161us vs triton 8100us (~7x, ~280ms/step at 40 layers).

Layout alignment (why this bridge is cheap)
-------------------------------------------
The stock face already stores qweights ``[E, N, K//2]`` u8 lo-even nibble --
exactly the kernel's uint32 view (no marlin weight shuffle; the shuffle
export exists but the kernel path takes raw, verified by parity). Prep is
one-time per weight tensor (cached via attr): channel scale [E,N,1] ->
repeat to the group-32 grid [E,N,K//32] -> bf16 -> ``w4a16_marlin_scale``
permutation (pmt_factor=4); w2 additionally K-padded as above. Activation
clamp uses the exact production formula silu(min(g,limit))*clamp(u,±limit)
via the tree's ``silu_and_mul_clamp`` op -- NOT aiter's ``"situ"``
(softcap beta1*tanh(g/b1)*sigmoid(g)... different function).

Guards (anything unmet -> next arm in the chain, log-once)
----------------------------------------------------------
* ``HCU_W4A16_MOE_IMPL=moe_c`` explicitly set (auto keeps the incumbent);
* int4-w4a16 face, bf16 activations, no zero-points, not no_combine, not
  apply_router_weight_on_input, activation silu-family;
* group size g (= K//scale.shape[-1]) with g % 32 == 0 and K % 32 == 0;
  (K//32) % 16 == 0 for gemm1; N1 % 256 == 0 and N2 % 256 == 0 (MODE-4/2
  tiles); I >= 256 (I==256 rides the kernel's special case, else padded to
  a 512 multiple).
Capture-safe: no host syncs (npost stays a device tensor), prep runs on
the warmup pass before decode graph capture, the activation pad columns of
the persistent a2 buffer are zeroed once and never written after.

Knob: ``HCU_W4A16_MOE_IMPL = auto | triton | deepgemm | moe_c``; tile knobs
``HCU_MOEC_W4A16_M1`` / ``_M2`` / ``_BAM`` (defaults 4 / 2 / 16).
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger("hcu.adapter")

_MODE_MOEC = ("moe_c", "moec")
_CACHE_ATTR = "_hcu_moec_w4a16_cache"
_miss_log_once: set = set()
_engaged_log: set = set()
_BUFS: dict = {}
# zeros args are required-but-inert for the symmetric (q-8)*s convention
# (DTK's own test passes uninitialized memory) -- ONE allocation per shape
# shared by every layer; 40 layers x 27MB of per-layer zeros OOM'd the
# first boot attempt (2026-10-26)
_ZEROS: dict = {}


def _mode() -> str:
    from . import hcu_env

    v, _src = hcu_env.resolve("HCU_W4A16_MOE_IMPL")
    return (v or "auto").strip().lower()


def _dbg(reason: str):
    if os.environ.get("HCU_MOEC_W4A16_DEBUG", ""):
        LOG.warning("HCU moe_c w4a16: %s", reason)


def _tile(name: str, dflt: int) -> int:
    from . import hcu_env

    v, _src = hcu_env.resolve(f"HCU_MOEC_W4A16_{name}")
    try:
        return int(v) if v else dflt
    except ValueError:
        return dflt


def pad_intermediate(I: int) -> int:
    """Gemm2 K law: (K/32) % 16 == 0 (K=256 special-case passes too)."""
    if I == 256 or I % 512 == 0:
        return I
    return (I + 511) // 512 * 512


def route_ok(
    w1_shape, w2_shape, s1_shape, s2_shape, dtype
) -> tuple[bool, str]:
    """Pure-shape engagement predicate (unit-testable without aiter/GPU)."""
    if dtype != torch.bfloat16:
        return False, f"dtype={dtype}"
    E, N1, Kh = w1_shape[0], w1_shape[1], w1_shape[2] * 2
    N2, Ki = w2_shape[1], w2_shape[2] * 2
    I = N1 // 2
    if Kh % 32 != 0 or Ki % 32 != 0:
        return False, "K%32"
    g1 = Kh // s1_shape[-1] if s1_shape[-1] else 0
    g2 = Ki // s2_shape[-1] if s2_shape[-1] else 0
    if g1 < 32 or g1 % 32 or g2 < 32 or g2 % 32 or g1 > Kh or g2 > Ki:
        return False, f"group {g1}/{g2} not on the 32-grid"
    if (Kh // 32) % 16 != 0:
        return False, "gemm1 K/32%16 (wheel law)"
    if N1 % 256 != 0 or N2 % 256 != 0:
        return False, f"N%256 ({N1},{N2})"
    if I < 256:
        return False, f"I={I}<256"
    return True, ""


def _prep(w1, w2, w1_scale, w2_scale):
    """One-time per-tensor prep: group-32 bf16 shuffled scales (+ w2 K-pad).

    Returns (s1_shuffled, s2_shuffled, w2_effective, z1, z2). Channel or
    coarse-group checkpoint scales are expanded to the 32-grid by value
    replication -- numerically exact (same product per element).
    """
    cached = getattr(w1, _CACHE_ATTR, None)
    if cached is not None:
        return cached
    from aiter.ops.shuffle import w4a16_marlin_scale

    K = w1.shape[2] * 2
    I = w1.shape[1] // 2
    H = w2.shape[1]
    IP = pad_intermediate(I)

    def g32(scale, kfull):
        g = kfull // scale.shape[-1]
        s = (scale.repeat_interleave(g // 32, dim=-1).contiguous()
             if g != 32 else scale.contiguous())
        return s.to(torch.bfloat16)

    s1 = w4a16_marlin_scale(g32(w1_scale, K))
    if IP == I:
        s2 = w4a16_marlin_scale(g32(w2_scale, I))
        w2e = w2
    else:
        w2e = torch.full((w2.shape[0], H, IP // 2), 0x88, dtype=torch.uint8,
                         device=w2.device)
        w2e[:, :, : I // 2] = w2
        # best-effort storage swap: the layer's own tensor now IS the
        # padded canvas, dropping the 283MB original (x40 layers this is
        # the difference between +1.2GB and +12.6GB per rank). The stock
        # triton arm can no longer read these weights -- the wrapper
        # raises loudly instead of silently corrupting after commit.
        try:
            w2.set_(w2e)
        except Exception:
            try:
                w2.data = w2e
            except Exception:
                pass
        s2p = torch.ones(w2.shape[0], H, IP // 32, dtype=torch.bfloat16,
                         device=w2.device)
        s2p[:, :, : I // 32] = g32(w2_scale, I)
        s2 = w4a16_marlin_scale(s2p)
    E = w1.shape[0]
    zk1 = (E, w1.shape[1] // 2, K // 32)
    zk2 = (E, H // 2, IP // 32)
    for zk in (zk1, zk2):
        if zk not in _ZEROS:
            _ZEROS[zk] = torch.zeros(zk, dtype=torch.uint8,
                                      device=w1.device)
    z1, z2 = _ZEROS[zk1], _ZEROS[zk2]
    out = (s1, s2, w2e, z1, z2)
    try:
        setattr(w1, _CACHE_ATTR, out)
    except Exception:
        pass
    return out


def _buffers(rows, N1, IP, H, device):
    key = (rows, device.index if device.index is not None else 0)
    got = _BUFS.get(key)
    if got is not None and got[0].shape[1] == N1 and got[2].shape[1] == H:
        return got
    c1 = torch.empty(rows, N1, dtype=torch.bfloat16, device=device)
    # pad columns are zeroed ONCE and never written afterwards (the clamp
    # kernel only fills [:, :I]) -- capture replays keep them zero
    a2 = torch.zeros(rows, IP, dtype=torch.bfloat16, device=device)
    c3 = torch.empty(rows, H, dtype=torch.bfloat16, device=device)
    act = torch.empty(rows, N1 // 2, dtype=torch.bfloat16, device=device)
    _BUFS[key] = (c1, a2, c3, act)
    return _BUFS[key]


def _clamp_silu(x2d, out, limit):
    """exact production formula silu(min(g,lim))*clamp(u,±lim) [rows,2I]->[rows,I]"""
    if limit is None:
        g, u = x2d[:, : out.shape[1]], x2d[:, out.shape[1]:]
        out.copy_((torch.nn.functional.silu(g) * u).to(out.dtype))
        return out
    try:
        from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp

        silu_and_mul_clamp(x2d.contiguous(), out, float(limit))
    except Exception:
        g = x2d[:, : out.shape[1]].clamp(max=float(limit))
        u = x2d[:, out.shape[1]:].clamp(min=-float(limit), max=float(limit))
        out.copy_((torch.nn.functional.silu(g) * u).to(out.dtype))
    return out


def try_moec_w4a16(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output,
    moe_runner_config,
    **kwargs,
):
    """Run the moe_c vendor pipeline for this int4 call, or None to fall
    through to the next arm (channel rewrite -> deepgemm -> aiter -> stock)."""
    if not kwargs.get("use_int4_w4a16"):
        return None
    if kwargs.get("w1_zp") is not None or kwargs.get("w2_zp") is not None:
        _dbg("zero-points present")
        return None
    mrc = moe_runner_config
    if bool(getattr(mrc, "no_combine", False)):
        _dbg("no_combine")
        return None
    if bool(getattr(mrc, "apply_router_weight_on_input", False)):
        _dbg("apply_router_weight_on_input")
        return None
    act = getattr(mrc, "activation", "silu")
    if isinstance(act, str) and "gelu" in act.lower():
        _dbg(f"activation {act}")
        return None

    import aiter  # noqa: F401  (kernel attr access below)
    from aiter.fused_moe_c import moe_align_block_size

    topk_weights, topk_ids, _ = topk_output
    E, N1, _ = w1.shape
    H = w2.shape[1]
    I = N1 // 2
    K = w1.shape[2] * 2
    ok, why = route_ok(w1.shape, w2.shape,
                       kwargs.get("w1_scale").shape if kwargs.get(
                           "w1_scale") is not None else (0,),
                       kwargs.get("w2_scale").shape if kwargs.get(
                           "w2_scale") is not None else (0,),
                       hidden_states.dtype)
    if not ok:
        key = (E, N1, H, K, why)
        if key not in _miss_log_once:
            _miss_log_once.add(key)
            LOG.warning(
                "HCU moe_c w4a16: shape miss %s (log-once) -> next arm", key)
        return None

    limit = getattr(mrc, "gemm1_clamp_limit", None)
    if limit is None:
        limit = getattr(mrc, "swiglu_limit", None)

    s1, s2, w2e, z1, z2 = _prep(w1, w2, kwargs["w1_scale"], kwargs["w2_scale"])
    IP = w2e.shape[2] * 2
    M = hidden_states.shape[0]
    topk = topk_ids.shape[-1]
    rows = M * topk
    bam = _tile("BAM", 16)
    m1 = _tile("M1", 4)
    m2 = _tile("M2", 2)

    ti = topk_ids.reshape(-1).to(torch.int32)
    if not torch.cuda.is_current_stream_capturing():
        # eager passes only: content guard (graph capture replays routing,
        # the warmup pass has already validated the same call face)
        if int(((ti < 0) | (ti >= E)).sum()):
            _dbg("topk_ids out of local range")
            return None
    ti = ti.contiguous().view(M, topk)
    tw = topk_weights.reshape(-1).to(torch.float32).contiguous()

    sorted_ids, expert_ids, npost = moe_align_block_size(ti, bam, E, None)
    c1, a2, c3, act_buf = _buffers(rows, N1, IP, H, hidden_states.device)

    aiter.moe_c_moe_gemm_marlin_w4a16(
        hidden_states, w1.view(torch.uint32), c1, s1, z1, None,
        sorted_ids, expert_ids, npost, topk, m1, 1)
    _clamp_silu(c1, act_buf, limit)
    a2[:, :I].copy_(act_buf)
    aiter.moe_c_moe_gemm_marlin_w4a16(
        a2, w2e.view(torch.uint32), c3, s2, z2, tw,
        sorted_ids, expert_ids, npost, 1, m2, 1)
    out = c3.view(M, topk, H).sum(1)
    rs = getattr(mrc, "routed_scaling_factor", None)
    if rs is not None and float(rs) != 1.0:
        out = out * float(rs)
    if len(_engaged_log) < 3 or (E, N1, H, K) not in _engaged_log:
        _engaged_log.add((E, N1, H, K))
        LOG.warning(
            "HCU moe_c w4a16: ENGAGED E=%d N1=%d H=%d K=%d topk=%d IP=%d "
            "M=%d limit=%r modes=(%d,%d) bam=%d",
            E, N1, H, K, topk, IP, M, limit, m1, m2, bam)
    if bool(getattr(mrc, "inplace", False)):
        hidden_states.copy_(out)
        return hidden_states
    return out


def install():
    """Wrap the stock ``fused_experts`` chain (outermost arm, 10.7)."""
    if _mode() not in _MODE_MOEC:
        LOG.info("HCU moe_c w4a16 bridge: off (HCU_W4A16_MOE_IMPL=%s)",
                 _mode())
        return
    try:
        import aiter  # noqa: F401
        import aiter.fused_moe_c  # noqa: F401
        import aiter.ops.shuffle  # noqa: F401
    except ImportError:
        LOG.info("HCU moe_c w4a16 bridge: aiter not importable -> skip")
        return

    import importlib

    mod_name = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    pkg_name = "sglang.srt.layers.moe.moe_runner.triton_utils"
    try:
        fm = importlib.import_module(mod_name)
    except ImportError:
        LOG.info("HCU moe_c w4a16 bridge: %s not importable -> skip", mod_name)
        return

    current = getattr(fm, "fused_experts", None)
    if current is None or getattr(current, "_hcu_moec_w4a16", False):
        return

    def fused_experts_moec(
        hidden_states, w1, w2, topk_output, moe_runner_config, **kwargs
    ):
        committed = getattr(w1, _CACHE_ATTR, None) is not None
        try:
            out = try_moec_w4a16(
                hidden_states=hidden_states, w1=w1, w2=w2,
                topk_output=topk_output,
                moe_runner_config=moe_runner_config, **kwargs)
            if out is not None:
                return out
        except Exception:
            # re-read: _prep may have committed (and swapped w2) inside
            # this very call -- a stale pre-call snapshot would fall back
            # to stock on swapped weights = silent garbage
            if committed or getattr(w1, _CACHE_ATTR, None) is not None:
                # weights may already be swapped to the padded canvas --
                # the stock arm would silently produce garbage; fail loud
                LOG.exception("HCU moe_c w4a16: failure AFTER weight "
                              "commit; refusing silent fallback")
                raise
            # pre-commit misses never take the server down for a
            # vendor-kernel experiment
            LOG.warning("HCU moe_c w4a16: call failed -> next arm",
                        exc_info=True)
        return current(hidden_states, w1, w2, topk_output, moe_runner_config,
                       **kwargs)

    fused_experts_moec._hcu_moec_w4a16 = True
    fm.fused_experts = fused_experts_moec
    try:
        pkg = importlib.import_module(pkg_name)
        if getattr(pkg, "fused_experts", None) is current:
            pkg.fused_experts = fused_experts_moec
    except ImportError:
        pass
    LOG.warning(
        "HCU moe_c w4a16 bridge: installed outermost (HCU_W4A16_MOE_IMPL=%s)",
        _mode())
