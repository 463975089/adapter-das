# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GDN AITER bridges: prefill state, packed decode and MTP target verification.
(catalog "linear_attn"; §0.5 route tier-② vendor-first, tier-③ triton
stays the default until this graduates).

The upstream faces (sglang 0.5.21, ``sglang.kernels.ops.attention.fla``):

- ``chunk_gated_delta_rule_fwd_h`` (prefill state recursion, per-chunk
  HBM stream) — the heaviest prefill stage; aiter ships the CK
  ``chunk_gated_delta_rule_fwd_sglang_hip_blockdim64`` with the same
  (h, v_new) return contract and the same [B, NT, H, V, K] h layout
  (gfx93x wave64-tuned, blockdim64).
- ``fused_recurrent_gated_delta_rule_packed_decode`` (decode fused
  QKV+gating+recurrence) — aiter ships
  ``aiter_fused_recurrent_gated_delta_rule_packed_decode`` with the same
  args except: no state-pool index argument (gather/scatter on the Python
  side) and fp32 state in / state out.

What this bridge does (MECH_REBIND, two import sites per face):

- fwd_h: keep upstream cumsum + intra (k/w/u) and chunk_o on the upstream
  triton kernels (cheap, already correct); swap only the state recursion
  for the aiter CK kernel. Envelope-strided state pools are handled with
  the gather/compute/scatter dance the operator team's sglang integration
  uses (aiter's API has no stride_init_state).
- packed decode: gather the batch's state slots to a contiguous fp32
  buffer, run the aiter packed kernel, scatter states back to the pool
  (fp32 -> pool dtype) and return ``(out_view, ht)`` matching the
  upstream ``(out, final_state)`` contract. Gate math (softplus of
  ``-exp(A_log)*dt + dt_bias``) belongs to the kernel on both sides.

Gate: HCU_LINEAR_ATTN_GDN = auto|aiter|triton. auto now follows the adapter
ladder (aiter > triton): the CK route passed the 6/6 numerics gate + Qwen3.5
TP4 e2e, so it is tried first; import/install failure keeps upstream triton.
=triton forces the upstream fla kernels.

Target verify preserves the upstream function object and cached imports.
HCU_LINEAR_ATTN_GDN_VERIFY=auto chooses AITER only for the qualified stack,
gfx936/gfx938, BF16 packed TP2 H8/HV24, FP32 state/snapshots and draft4.
There is no fixed batch ceiling; the runtime retains its resource/shape limits.
Other contracts stay upstream; =triton and HCU_DISABLE_OPS=gdn_verify disable it.

Prefill/packed decode gate: HCU_LINEAR_ATTN_GDN = auto|aiter|triton (auto = triton until the
numerics gate + e2e row graduate this to candidate/production).

"""

import inspect
from importlib import metadata
import logging
import types

import torch

LOG = logging.getLogger("hcu.adapter")


def _gather_states(pool, indices):
    """Contiguous fp32 copy of the batch's state slots (aiter has no
    state-pool index argument and wants fp32)."""
    idx = indices.clamp_min(0).to(torch.long)
    local = torch.arange(indices.numel(), device=indices.device,
                         dtype=indices.dtype)
    states = pool.index_select(0, idx).to(dtype=torch.float32).contiguous()
    return states, local


def _scatter_states(pool, indices, local_indices, states):
    valid = indices >= 0
    if valid.any():
        pool.index_copy_(
            0, indices[valid].to(torch.long),
            states[valid].to(dtype=pool.dtype),
        )


def _install_fwd_h_aiter(registry):
    """Swap sglang.kernels...fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h
    for the aiter CK state recursion; rebind the chunk.py consumer too."""
    # 替代档: aiter CK（本文件 rebind 宿主 fla 档）
    from sglang.kernels.ops.attention.fla import chunk_delta_h as _cdh
    # 替代档: aiter CK（本文件 rebind 宿主 fla 档）
    from sglang.kernels.ops.attention.fla import chunk as _chunk_mod

    _orig_fwd_h = _cdh.chunk_gated_delta_rule_fwd_h

    def chunk_gated_delta_rule_fwd_h(
        k, w, u, g=None, gk=None, initial_state=None,
        initial_state_indices=None, save_new_value=True,
        cu_seqlens=None, chunk_indices=None, use_exp2=False,
        inplace_update=True, track_state=None, track_chunk_idx=None,
    ):
        # aiter instantiates K=V=128 BT=64 (the GDN production shape);
        # anything else keeps the upstream triton kernel.
        if gk is not None or track_state is not None or use_exp2:
            return _orig_fwd_h(
                k, w, u, g, gk, initial_state, initial_state_indices,
                save_new_value, cu_seqlens, chunk_indices, use_exp2,
                inplace_update, track_state, track_chunk_idx,
            )
        B, T, Hg, K = k.shape
        H, V = u.shape[-2], u.shape[-1]
        if K != 128 or V != 128:
            return _orig_fwd_h(
                k, w, u, g, gk, initial_state, initial_state_indices,
                save_new_value, cu_seqlens, chunk_indices, use_exp2,
                inplace_update, track_state, track_chunk_idx,
            )
        import aiter.ops.fla as _afla

        if chunk_indices is None and cu_seqlens is not None:
            # 替代档: 宿主 fla helper（本文件 aiter 链路复用）
            from sglang.kernels.ops.attention.fla.index import (
                prepare_chunk_indices,
            )

            chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

        aiter_state = initial_state
        aiter_idx = initial_state_indices
        valid_mask = None
        if initial_state is not None and initial_state_indices is not None:
            valid_mask = initial_state_indices >= 0
            aiter_state, aiter_idx = _gather_states(
                initial_state, initial_state_indices)

        h, v_new = _afla.chunk_gated_delta_rule_fwd_sglang_hip_blockdim64(
            k=k, w=w, u=u, g=g, gk=None,
            initial_state=aiter_state,
            initial_state_indices=aiter_idx,
            output_final_state=True,
            chunk_size=64,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            transpose_state_layout=True,
        )
        if valid_mask is not None:
            _scatter_states(
                initial_state, initial_state_indices, aiter_idx, aiter_state)
        return h, v_new

    for ns in (_cdh, _chunk_mod):
        pass  # chunk.py imports fwd_h at module top; see rebind below
    _cdh.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    # chunk.py bound the original symbol at import time (from-import);
    # the is_intel branch absent on HCU means its module-global
    # chunk_gated_delta_rule_fwd_h IS the chunk_delta_h one — rebind there
    # as well so ChunkGatedDeltaRuleFunction calls ours.
    import sys

    already = sys.modules.get("sglang.kernels.ops.attention.fla.chunk")
    if already is not None:
        already.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    registry.register(
        "linear_attn_gdn",
        chunk_gated_delta_rule_fwd_h,
        target="sglang.kernels...fla.chunk_delta_h:chunk_gated_delta_rule_fwd_h",
        mechanism=registry.MECH_REBIND,
        detail="state recursion (fwd_h) -> aiter CK blockdim64; K=V=128 only, "
               "rest keeps upstream triton; state pool gather/scatter wrap",
    )
    LOG.warning(
        "HCU adapter: GDN prefill state recursion routed to aiter CK "
        "(chunk_gated_delta_rule_fwd_h; K=V=128, gather/scatter state pool)"
    )


def _install_packed_decode_aiter(registry):
    """Swap the packed decode fused kernel for aiter's triton fla variant
    (same contract incl. state-pool indices — the operator team's own
    sglang alignment; drop-in rebind)."""
    # 替代档: aiter triton fla（本文件 packed-decode rebind）
    from sglang.kernels.ops.attention.fla import fused_recurrent as _fr
    from aiter.ops.triton.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as _aiter_decode,
    )

    _fr.fused_recurrent_gated_delta_rule_packed_decode = _aiter_decode
    registry.register(
        "linear_attn_gdn",
        _aiter_decode,
        target="sglang.kernels...fla.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
        mechanism=registry.MECH_REBIND,
        detail="packed decode -> aiter triton fla (same contract, "
               "state-pool indices native)",
    )
    LOG.warning(
        "HCU adapter: GDN packed decode routed to aiter triton fla kernel "
        "(same contract drop-in)"
    )


# Target verification: guarded AITER route with cached aliases preserved.
_VERIFY_MARKER = "_hcu_gdn_verify_aiter_original"
_VERIFY_DISPATCH = "_HCU_GDN_VERIFY_AITER_DISPATCH"
_VERIFY_POLICY = "_hcu_gdn_verify_aiter_policy"
# Qualification is tied to the measured build. New stacks stay upstream until
# their eager/Graph/reference/model tests extend this admission table.
_VERIFY_VALIDATED_STACK = {
    "sglang": "0.5.22",
    "torch": "2.11.0+dtk2610.2611031312.g529def",
    "triton": "3.6.0+dtk2610.torch2110.2611031429.g92771c",
    "aiter": "0.1.8+dtk2610.torch2110.2611031115.ga1a2bc",
}

_VERIFY_AUTO_ARCHES = frozenset(("gfx936", "gfx938"))

_VERIFY_REQUIRED = frozenset((
    "A_log", "a", "dt_bias", "softplus_beta", "softplus_threshold",
    "q", "k", "v", "b", "initial_state_source", "initial_state_indices",
    "disable_state_update", "intermediate_states_buffer",
    "intermediate_state_indices", "cache_steps", "retrieve_parent_token",
))
_VERIFY_UNSUPPORTED = (
    "cache_ring", "match_cutedsl_decode", "replayssm_rawv", "replayssm_rawk",
    "replayssm_g", "replayssm_beta",
)


def _verify_auto_stack_supported():
    """Check distribution metadata only; never initialize/probe a GPU here."""
    try:
        return all(metadata.version(name) == value
                   for name, value in _VERIFY_VALIDATED_STACK.items())
    except (metadata.PackageNotFoundError, ValueError):
        return False


def _verify_auto_contract_supported(p, torch):
    """The verified packed TP2 GDN profile; all checks are tensor metadata."""
    cu = p.get("cu_seqlens")
    if cu is None:
        return False
    n = cu.numel() - 1
    if (n < 1 or tuple(p["q"].shape[:3]) != (1, n * 4, 8)
            or p["v"].shape[-2] != 24 or p["cache_steps"] != 4
            or p.get("use_qk_l2norm_in_kernel") is not True
            or p.get("lower_bound") is not None):
        return False
    for name, expected in (("softplus_beta", 1.0), ("softplus_threshold", 20.0)):
        value = p[name]
        if type(value) not in (int, float) or value != expected:
            return False
    scale = p.get("scale")
    if scale is not None and (type(scale) not in (int, float) or scale != 128**-0.5):
        return False
    if any(p[name].dtype not in (torch.bfloat16, torch.float32)
           for name in ("a", "b")):
        return False
    return (p["A_log"].dtype == torch.float32
            and p["dt_bias"].dtype in (torch.float32, torch.bfloat16))


def _verify_entry(*args, **kwargs):
    # Executed in the upstream function's globals, not this module's.
    return _HCU_GDN_VERIFY_AITER_DISPATCH(*args, **kwargs)


def _verify_contract_supported(p, torch):
    if p.get("disable_state_update") is not True or p.get("is_kda", False) is not False:
        return False
    if any(p.get(name) is not None and p.get(name) is not False
           for name in _VERIFY_UNSUPPORTED):
        return False
    try:
        q, k, v = p["q"], p["k"], p["v"]
        if q.ndim != 4 or q.shape != k.shape or v.ndim != 4:
            return False
        batch, tokens, heads, width = q.shape
        hv, value_width = v.shape[-2:]
        if batch <= 0 or tokens <= 0 or width != 128 or value_width != 128:
            return False
        if heads <= 0 or hv <= 0 or hv % heads or v.shape[:2] != q.shape[:2]:
            return False
        for x, dim in ((q, width), (k, width), (v, value_width)):
            if (x.device.type != "cuda" or x.dtype != torch.bfloat16
                    or x.stride(-1) != 1 or x.stride(-2) != dim):
                return False
            if batch > 1 and x.stride(0) != tokens * x.stride(1):
                return False
        steps = p["cache_steps"]
        if not isinstance(steps, int) or steps <= 0:
            return False
        state, scratch = p["initial_state_source"], p["intermediate_states_buffer"]
        if (state is None or scratch is None or state.ndim != 4 or scratch.ndim != 5
                or tuple(state.shape[1:]) != (hv, value_width, width)
                or tuple(scratch.shape[1:]) != (steps, hv, value_width, width)):
            return False
        if any(x.dtype != torch.float32 or not x.is_contiguous()
               for x in (state, scratch)):
            return False
        tensors = [q, k, v, state, scratch, p["a"], p["b"],
                   p["A_log"], p["dt_bias"], p["initial_state_indices"],
                   p["intermediate_state_indices"]]
        tensors += [p[n] for n in ("cu_seqlens", "retrieve_parent_token")
                    if p.get(n) is not None]
        if any(x.device != q.device for x in tensors):
            return False
        if p["a"].numel() != batch * tokens * hv or p["b"].numel() != batch * tokens * hv:
            return False
        # SGLang target_verify passes [B*T, HV]; the public face also
        # accepts the explicit [B, T, HV] gate layout. Both flatten to
        # the same token/head contract consumed by AITER.
        if tuple(p["b"].shape) not in ((batch * tokens, hv), (batch, tokens, hv)):
            return False
        if p["A_log"].numel() != hv or p["dt_bias"].numel() != hv:
            return False
        if not p["A_log"].is_contiguous() or not p["dt_bias"].is_contiguous():
            return False
        n = batch if p.get("cu_seqlens") is None else p["cu_seqlens"].numel() - 1
        if n <= 0:
            return False
        for name in ("initial_state_indices", "intermediate_state_indices"):
            idx = p[name]
            if (idx.ndim != 1 or idx.numel() < n or idx.stride(0) != 1
                    or idx.dtype not in (torch.int32, torch.int64)):
                return False
        cu = p.get("cu_seqlens")
        if cu is not None and (cu.ndim != 1 or cu.stride(0) != 1
                               or cu.dtype not in (torch.int32, torch.int64)):
            return False
        parents = p.get("retrieve_parent_token")
        if parents is not None and (parents.ndim != 2 or parents.shape[0] < n
                                    or parents.shape[1] < steps
                                    or parents.dtype not in (torch.int32, torch.int64)):
            return False
        return True
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _bind_target_verify_aiter(function, vendor, torch, *, policy="aiter"):
    if getattr(function, _VERIFY_MARKER, None) is not None:
        return function
    if not inspect.isfunction(function) or function.__closure__:
        raise RuntimeError("GDN verify: unsupported upstream callable/closure")
    signature = inspect.signature(function)
    vendor_signature = inspect.signature(vendor)
    if not _VERIFY_REQUIRED.issubset(signature.parameters):
        raise RuntimeError("GDN verify: upstream contract changed")
    if not _VERIFY_REQUIRED.issubset(vendor_signature.parameters):
        raise RuntimeError("GDN verify: AITER verify contract unavailable")
    if _VERIFY_DISPATCH in function.__globals__:
        raise RuntimeError("GDN verify: foreign dispatch binding")
    original = types.FunctionType(
        function.__code__, function.__globals__, function.__name__,
        function.__defaults__, function.__closure__,
    )
    original.__kwdefaults__ = function.__kwdefaults__
    vendor_names = frozenset(vendor_signature.parameters)
    known = vendor_names | frozenset(_VERIFY_UNSUPPORTED)
    if set(signature.parameters) - known:
        raise RuntimeError("GDN verify: unreviewed upstream parameter")
    announced = False
    declined = False
    device_arches = {}
    device_probe_errors = {}

    def dispatch(*args, **kwargs):
        nonlocal announced, declined
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return original(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        supported = _verify_contract_supported(p, torch)
        base_supported = supported
        auto_supported = None
        if supported and policy == "auto":
            supported = _verify_auto_contract_supported(p, torch)
            auto_supported = supported
            if supported:
                device = p["q"].device
                if device.index not in device_arches:
                    try:
                        # The input already exists on its assigned device.
                        # Explicit device lookup avoids default-GPU bootstrap.
                        prop = torch.cuda.get_device_properties(device)
                        device_arches[device.index] = prop.gcnArchName.split(":")[0]
                    except (AttributeError, RuntimeError, ValueError) as exc:
                        device_arches[device.index] = ""
                        device_probe_errors[device.index] = type(exc).__name__
                supported = device_arches[device.index] in _VERIFY_AUTO_ARCHES
        if not supported:
            if (not declined and p.get("disable_state_update") is True
                    and p.get("is_kda", False) is False):
                q_device = getattr(p.get("q"), "device", None)
                device_index = getattr(q_device, "index", None)
                def scalar(value):
                    if type(value) in (int, float, bool, str, type(None)):
                        return value
                    return type(value).__name__

                info = {
                    "policy": policy, "base_contract": base_supported,
                    "auto_contract": auto_supported,
                    "arch": device_arches.get(device_index, "not queried"),
                    "arch_probe_error": device_probe_errors.get(device_index),
                    "tree_parents": p.get("retrieve_parent_token") is not None,
                    **{name: scalar(p.get(name)) for name in
                       ("cache_steps", "scale", "use_qk_l2norm_in_kernel",
                        "softplus_beta", "softplus_threshold", "lower_bound")},
                }
                for name in ("q", "a", "b", "A_log", "dt_bias", "initial_state_source",
                             "intermediate_states_buffer"):
                    value = p.get(name)
                    if value is not None:
                        info[name] = (tuple(value.shape), str(value.dtype), value.stride())
                log = LOG.debug if policy == "auto" else LOG.warning
                log("HCU adapter: GDN verify retained upstream for unsupported "
                    "contract: %s", info)
                declined = True
            return original(*args, **kwargs)
        forwarded = {name: value for name, value in p.items() if name in vendor_names}
        # AITER has token strides for q/k/v but needs a dense beta head vector.
        if not forwarded["b"].is_contiguous():
            forwarded["b"] = forwarded["b"].contiguous()
        if not announced:
            LOG.warning("HCU adapter: GDN target_verify executing AITER "
                        "(%s; policy=%s, arch=%s)", vendor.__module__, policy,
                        device_arches.get(p["q"].device.index, "explicit"))
            announced = True
        # A provider failure may follow a snapshot write; never silently retry.
        return vendor(**forwarded)

    function.__globals__[_VERIFY_DISPATCH] = dispatch
    function.__signature__ = signature
    setattr(function, _VERIFY_MARKER, original)
    setattr(function, _VERIFY_POLICY, policy)
    function.__code__ = _verify_entry.__code__
    return function


def _load_target_verify_providers():
    import torch
    if torch.version.hip is None:
        return None
    # 替代档: 宿主 fla sigmoid-gating 面（target-verify 对拍与回退的契约锚）
    from sglang.kernels.ops.attention.fla import fused_sigmoid_gating_recurrent as face
    from aiter.ops.triton.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as vendor,
    )
    return torch, face.fused_sigmoid_gating_delta_rule_update, vendor


def _install_target_verify_aiter():
    from hcu_adapter import hcu_env
    from hcu_adapter.kill_switch import op_disabled
    if op_disabled("gdn_verify"):
        LOG.warning("HCU adapter: GDN verify kept upstream by HCU_DISABLE_OPS=gdn_verify")
        return
    mode, source = hcu_env.resolve("HCU_LINEAR_ATTN_GDN_VERIFY")
    if mode not in ("auto", "triton", "aiter"):
        raise ValueError("HCU_LINEAR_ATTN_GDN_VERIFY must be auto, triton or aiter")
    if mode == "triton":
        return
    if mode == "auto" and not _verify_auto_stack_supported():
        LOG.debug("HCU adapter: GDN verify auto kept upstream: stack not qualified")
        return
    try:
        providers = _load_target_verify_providers()
        if providers is None:
            return
        torch, function, vendor = providers
        _bind_target_verify_aiter(function, vendor, torch, policy=mode)
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError):
        if mode == "aiter":
            raise
        LOG.warning("HCU adapter: GDN verify auto kept upstream: provider/API "
                    "unavailable", exc_info=True)
        return
    from hcu_adapter import registry
    registry.register(
        "linear_attn_gdn_verify", vendor,
        target=("sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent."
                "fused_sigmoid_gating_delta_rule_update"),
        mechanism=registry.MECH_REBIND,
        detail="policy=" + mode + "; BF16 target_verify; FP32 state/snapshots; "
               "auto admits qualified gfx936/gfx938, TP2 H8/HV24 draft4; no fixed batch ceiling; "
               "cached aliases retained; other contracts keep upstream",
    )
    LOG.warning("HCU adapter: GDN verify AITER route installed "
                "(HCU_LINEAR_ATTN_GDN_VERIFY=%s, source=%s)", mode, source)


def install():
    from hcu_adapter import hcu_env
    # Verify auto policy is independent of the prefill/packed decode selector.
    _install_target_verify_aiter()

    mode, _src = hcu_env.resolve("HCU_LINEAR_ATTN_GDN")
    if mode == "triton":
        LOG.warning(
            "HCU adapter: HCU_LINEAR_ATTN_GDN=triton -- GDN stays on the "
            "upstream triton fla kernels (aiter route available via =aiter)",
        )
        return
    # auto (new default) and aiter both take the validated CK route; any
    # import/install failure leaves the upstream triton fla faces intact.
    try:
        import aiter  # noqa: F401
        import aiter.ops.fla  # noqa: F401
    except Exception:
        if mode == "aiter":
            LOG.warning(
                "HCU adapter: HCU_LINEAR_ATTN_GDN=aiter but aiter is not "
                "importable; GDN stays on triton fla",
                exc_info=True,
            )
        else:
            LOG.info(
                "HCU adapter: GDN auto: aiter not importable; stays on "
                "upstream triton fla",
            )
        return

    from hcu_adapter import registry

    try:
        _install_fwd_h_aiter(registry)
        _install_packed_decode_aiter(registry)
    except Exception:
        LOG.warning(
            "HCU adapter: GDN aiter install failed; keeping upstream triton "
            "fla faces",
            exc_info=True,
        )
