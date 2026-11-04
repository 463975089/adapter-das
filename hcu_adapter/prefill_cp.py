# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU prefill context-parallel (CP) bridge.

sglang v0.5.21 ships full prefill-CP machinery (``srt/layers/cp``:
zigzag / interleave strategies, attn-cp process group, fa3-family
attention dispatch convention) and wires it into qwen3_moe / qwen3_5 /
deepseek_v2/v4 et al., but ``arg_groups/parallel_hook.py::
validate_prefill_cp_platform`` rejects HIP outright ("Prefill CP on
HIP/MUSA is deprecated; CP support will be refactored soon").

This bridge has four pieces, all behind an explicit opt-in knob:

    HCU_PREFILL_CP=1  (default 0 = stock rejection kept)

1. Platform gate: replace validate_prefill_cp_platform so the knob lets
   prefill CP start on HIP. Knob off -> upstream rejection preserved.

2. CP rope guard: on HIP the fused rope+KV-write kernel
   (``fused_qk_rope_reshape_and_cache``, the only rope path on HIP that
   writes K/V) asserts ``t == t_slot``. Under CP chunked extend the
   local token count t is SMALLER than the full-sequence slot count
   t_slot (zigzag: 6-token seq -> rank0 owns 4 local q/k/v but
   out_cache_loc still covers all 6), so the assert fires on the very
   first prefill. Guard: wrap every rotary ``forward_cuda`` that takes
   ``fused_set_kv_buffer_arg``; when slots != tokens (exactly the CP
   chunked case, self-detected -- no global state needed), drop the
   fused arg so rope falls back to the plain kernel and the KV write is
   left to the attention backend / CP strategy -- the same flow CUDA CP
   uses. Decode keeps t == t_slot so the fused fast path is untouched.

3. enable_fused_set_kv_buffer CP guard (root cause of empty-KV): the
   upstream predicate's HIP arm lacks the ``not is_cp_active`` check the
   CUDA arm has, so under CP models believe the fused rope will write
   the KV pool and pass save_kv_cache=False -- with piece 2 dropping
   the fused write, nobody writes KV and every CP prefill attends over
   an empty pool (garbage output). Wrapping the predicate to return
   False under CP makes the fa3 backend run
   ``cp_strategy.materialize_full_kv`` -- the CUDA CP flow.

4. Optional instrumentation (HCU_PREFILL_CP_DEBUG=1): checksum-logging
   wrappers on the zigzag strategy stages + fa3 emu calls.

Launch recipe (validated on nmz26 gfx938, fa3 emu stack):
    --tp-size 2 --attn-cp-size 2 --enable-prefill-cp --cp-strategy zigzag
    --attention-backend fa3

Everything else (strategy init, attn-cp group NCCL comm, zigzag KV
exchange, decode CP attention) is upstream code running unmodified.
"""

from __future__ import annotations

import logging

LOG = logging.getLogger("hcu.adapter")

_TARGET_MOD = "sglang.srt.arg_groups.parallel_hook"
_ROPE_MOD = "sglang.srt.layers.rotary_embedding.base"

_drop_logged = False


def validate_prefill_cp_platform_hcu(server_args):
    """Replacement for the stock validator.

    HCU_PREFILL_CP=1 -> allow prefill CP on HIP (experimental).
    Otherwise behave exactly like upstream: reject on HIP/MUSA.
    """
    from . import hcu_env

    knob, src = hcu_env.resolve("HCU_PREFILL_CP")
    if knob not in ("", "0", "None"):
        LOG.warning(
            "HCU prefill-CP bridge: platform gate opened (%s=%s, %s) -- "
            "upstream marks HIP CP deprecated; treat as experimental",
            "HCU_PREFILL_CP", knob, src,
        )
        return None

    # stock semantics (kept verbatim for the knob-off path)
    cfg = server_args  # resolving_view handles attr lookup; keep it simple
    try:
        from sglang.srt.runtime_context import get_platform

        platform = get_platform()
        enable_prefill_cp = getattr(cfg, "enable_prefill_cp", False)
        if enable_prefill_cp and (platform.is_hip or platform.is_musa):
            raise ValueError(
                "Prefill CP on HIP/MUSA is deprecated; CP support will "
                "be refactored soon."
            )
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# piece 2: CP rope guard
# ---------------------------------------------------------------------------

def _wrap_forward_cuda(orig):
    """Wrap a rotary forward_cuda: drop fused_set_kv_buffer_arg when the
    slot count disagrees with the local token count (CP chunked extend).

    Self-gating on the exact invariant the fused kernel asserts means
    non-CP paths (and CP decode, where t == t_slot) keep the fused fast
    path with zero global state.
    """

    def forward_cuda(self, *args, **kwargs):
        global _drop_logged

        fused = kwargs.get("fused_set_kv_buffer_arg")
        if fused is not None and args:
            q = args[1] if len(args) > 1 else None
            if q is None or not hasattr(q, "shape") or q.dim() == 0:
                return orig(self, *args, **kwargs)
            slots = (
                fused.get("slot_mapping")
                if isinstance(fused, dict)
                else getattr(fused, "cache_loc", None)
            )
            if slots is not None and int(slots.numel()) != int(q.shape[0]):
                kwargs["fused_set_kv_buffer_arg"] = None
                if not _drop_logged:
                    _drop_logged = True
                    LOG.warning(
                        "HCU prefill-CP bridge: CP chunked extend detected "
                        "(tokens=%d, slots=%d) -> fused rope+KV-write "
                        "skipped, plain rope + backend KV write (CUDA-CP "
                        "flow); logged once",
                        int(q.shape[0]), int(slots.numel()),
                    )
        return orig(self, *args, **kwargs)

    return forward_cuda


def _patch_rotary_module(mod):
    """Wrap forward_cuda on every rotary class in `mod` that accepts
    fused_set_kv_buffer_arg (subclasses inherit the wrapper unless they
    override forward_cuda themselves; walk all defs to be safe)."""
    import inspect

    if getattr(mod, "_HCU_CP_ROPE_PATCHED", False):
        return
    patched = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            continue
        f = obj.__dict__.get("forward_cuda")
        if f is None or not callable(f):
            continue
        try:
            params = inspect.signature(f).parameters
        except (ValueError, TypeError):
            continue
        if "fused_set_kv_buffer_arg" not in params:
            continue
        setattr(obj, "forward_cuda", _wrap_forward_cuda(f))
        patched.append(name)
    mod._HCU_CP_ROPE_PATCHED = True
    if patched:
        LOG.warning(
            "HCU prefill-CP bridge: rope guard wrapped forward_cuda on %s",
            patched,
        )


# ---------------------------------------------------------------------------
# piece 3: CP stage instrumentation (HCU_PREFILL_CP_DEBUG=1)
# ---------------------------------------------------------------------------

def _dbg_sum(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return f"{tuple(x.shape)}:{float(x.float().sum()):.3f}"
    except Exception:
        pass
    return type(x).__name__


def _patch_zigzag_module(mod):
    """Wrap ZigzagContextParallelStrategy stage methods with checksum logs so
    the first numerically-divergent CP stage shows up in the server log."""
    import os

    if getattr(mod, "_HCU_CP_DBG_PATCHED", False):
        return
    if os.environ.get("HCU_PREFILL_CP_DEBUG", "") == "":
        mod._HCU_CP_DBG_PATCHED = True
        return

    strat = getattr(mod, "ZigzagCPStrategy", None)
    if strat is None:
        return

    def _wrap(orig, label, arg_idx, res_idx):
        def wrapped(self, *args, **kwargs):
            pre = ",".join(_dbg_sum(a) for a in args[arg_idx[0] : arg_idx[1]])
            out = orig(self, *args, **kwargs)
            res = _dbg_sum(out) if res_idx is None else _dbg_sum(out[res_idx])
            LOG.warning("CPDBG[%s r%s] in(%s) out(%s)", label, self.cp_rank, pre, res)
            return out

        return wrapped

    # (method, first positional arg to checksum, last+1; result index or None)
    for mname, a0, a1 in (
        ("shard_hidden_states", 0, 1),
        ("shard_position_ids", 0, 1),
        ("gather_kv_cache", 0, 1),
        ("gather_hidden_states", 0, 1),
        ("materialize_full_kv", 2, 4),  # k, v
    ):
        fn = getattr(strat, mname, None)
        if fn is None:
            continue
        setattr(strat, mname, _wrap(fn, mname, (a0, a1), None))
    mod._HCU_CP_DBG_PATCHED = True
    LOG.warning("HCU prefill-CP bridge: CPDBG instrumentation active")


# ---------------------------------------------------------------------------
# piece 4: enable_fused_set_kv_buffer CP guard (THE root-cause fix)
# ---------------------------------------------------------------------------

_MODELS_UTILS_MOD = "sglang.srt.models.utils"


def _patch_models_utils(mod):
    """Upstream models/utils.py enable_fused_set_kv_buffer:

        (_is_cuda and ... and not is_cp_active(forward_batch) and ...)
        or (_is_hip and dcp_kv_mask is None)

    The HIP arm is missing the CP guard. Under prefill CP the HIP predicate
    returns True, so models skip passing save_kv_cache to the attention
    backend (qwen3_moe forward_core: save_kv_cache = ... or not
    enable_fused_set_kv_buffer(...)), believing the fused rope wrote the KV
    pool. With the rope guard (piece 2) the fused write is dropped, so
    NOBODY writes KV: the backend's materialize_full_kv (its cp_active
    branch) is the intended writer and only runs when save_kv_cache=True --
    exactly the CUDA CP flow. Wrapping the predicate to return False under
    CP restores that flow on HIP.
    """
    import inspect

    if getattr(mod, "_HCU_CP_FUSEKV_PATCHED", False):
        return
    orig = getattr(mod, "enable_fused_set_kv_buffer", None)
    if orig is None or not callable(orig):
        return
    try:
        if "forward_batch" not in inspect.signature(orig).parameters:
            return
    except (ValueError, TypeError):
        return

    def enable_fused_set_kv_buffer_hcu(forward_batch):
        from sglang.srt.layers.cp.utils import is_cp_active

        try:
            if is_cp_active(forward_batch):
                return False
        except Exception:
            pass
        return orig(forward_batch)

    mod.enable_fused_set_kv_buffer = enable_fused_set_kv_buffer_hcu
    mod._HCU_CP_FUSEKV_PATCHED = True
    LOG.warning(
        "HCU prefill-CP bridge: enable_fused_set_kv_buffer CP guard patched"
    )


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def install():
    """Patch the platform gate (always; knob checked at call time) and the
    rope guard (only when HCU_PREFILL_CP resolves on)."""
    import sys

    from . import hcu_env
    from ._base import LibStubFinder, register_finder

    # piece 1: platform gate
    mod = sys.modules.get(_TARGET_MOD)
    if mod is not None:
        _patch(mod)
    else:
        # extras only fill missing attrs; the stock module already defines
        # validate_prefill_cp_platform, so this must be an override
        register_finder(
            LibStubFinder(
                _TARGET_MOD,
                overrides={
                    _TARGET_MOD: {
                        "validate_prefill_cp_platform":
                            validate_prefill_cp_platform_hcu,
                        "HCU_PREFILL_CP_BRIDGE": True,
                    }
                },
            )
        )
        LOG.warning("HCU prefill-CP bridge: platform gate patch registered")

    # piece 2: rope guard (behavioral change -- knob-gated)
    knob, src = hcu_env.resolve("HCU_PREFILL_CP")
    if knob in ("", "0", "None"):
        return
    rope_mod = sys.modules.get(_ROPE_MOD)
    if rope_mod is not None:
        _patch_rotary_module(rope_mod)
    else:
        register_finder(
            LibStubFinder(
                _ROPE_MOD,
                post_hooks={_ROPE_MOD: _patch_rotary_module},
            )
        )
        LOG.warning(
            "HCU prefill-CP bridge: rope guard registered (%s=%s, %s)",
            "HCU_PREFILL_CP", knob, src,
        )

    # piece 3: debug instrumentation (HCU_PREFILL_CP_DEBUG=1)
    _ZIGZAG_MOD = "sglang.srt.layers.cp.zigzag"
    zz = sys.modules.get(_ZIGZAG_MOD)
    if zz is not None:
        _patch_zigzag_module(zz)
    else:
        register_finder(
            LibStubFinder(
                _ZIGZAG_MOD,
                post_hooks={_ZIGZAG_MOD: _patch_zigzag_module},
            )
        )

    # piece 4: enable_fused_set_kv_buffer CP guard (root-cause fix; must be
    # a post-load patch so models importing the symbol BY NAME bind the
    # wrapper -- overrides/extras cannot express "wrap the real fn")
    mu = sys.modules.get(_MODELS_UTILS_MOD)
    if mu is not None:
        _patch_models_utils(mu)
    else:
        register_finder(
            LibStubFinder(
                _MODELS_UTILS_MOD,
                post_hooks={_MODELS_UTILS_MOD: _patch_models_utils},
            )
        )


def _patch(mod):
    if getattr(mod, "HCU_PREFILL_CP_BRIDGE", False):
        return
    mod.validate_prefill_cp_platform = validate_prefill_cp_platform_hcu
    mod.HCU_PREFILL_CP_BRIDGE = True
    LOG.info("HCU prefill-CP bridge: platform gate patched (module already loaded)")
