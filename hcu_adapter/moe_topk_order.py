# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Keep native grouped routing order stable without changing its selection.

HIP's unsorted Torch top-k can permute the same selected experts between
calls. Expert-slot order then changes the finite-precision MoE reduction.
Only two validated sorted operands change in a private GPU function copy.
CPU routing, bias, grouping, normalization and shared-expert handling remain
upstream. Function objects and the compiled wrapper stay in place.
"""

from __future__ import annotations

import dis
import inspect
import logging
import threading
import types

from . import registry as _reg

_LOCK = threading.RLock()
_DISPATCH = "_hcu_ordered_native_topk_impl"
_STOCK = "_hcu_stock_native_topk_impl"
_MARKER = "_hcu_ordered_native_topk_codes"
_PARAMETERS = (
    "hidden_states", "gating_output", "correction_bias", "topk",
    "renormalize", "num_expert_group", "topk_group",
    "num_fused_shared_experts", "routed_scaling_factor",
    "apply_routed_scaling_factor_on_output",
)
LOG = logging.getLogger("hcu.adapter")


def _ordered_code(code):
    instructions = list(dis.get_instructions(code))
    sites = [i for i, op in enumerate(instructions)
             if op.opname == "LOAD_CONST" and op.argval is False]
    if len(sites) != 2:
        raise RuntimeError("moe_topk_order: native sorted operands changed")
    for i in sites:
        # Python 3.10 uses a keyword-name tuple + CALL_FUNCTION_KW;
        # newer Python uses KW_NAMES + PRECALL/CALL.
        following = instructions[i + 1]
        if following.opname == "LOAD_CONST":
            keywords = following.argval
        elif following.opname == "KW_NAMES":
            keywords = code.co_consts[following.arg]
        else:
            raise RuntimeError("moe_topk_order: sorted is no longer a keyword operand")
        call = instructions[i + 2]
        if (keywords != ("k", "dim", "sorted")
                or call.opname not in ("CALL_FUNCTION_KW", "PRECALL", "CALL")
                or call.arg != 4):
            raise RuntimeError("moe_topk_order: upstream topk arguments changed")
    return code.replace(co_consts=tuple(
        True if value is False else value for value in code.co_consts
    ))


def _native(hidden_states, gating_output, correction_bias, topk, renormalize,
            num_expert_group, topk_group, num_fused_shared_experts,
            routed_scaling_factor, apply_routed_scaling_factor_on_output):
    # Retain the native frame's argument names/count as well as its defaults.
    # Dynamo can reuse an existing wrapper after this function is installed.
    implementation = (_hcu_ordered_native_topk_impl
                      if gating_output.device.type == "cuda"
                      else _hcu_stock_native_topk_impl)
    return implementation(
        hidden_states, gating_output, correction_bias, topk, renormalize,
        num_expert_group, topk_group, num_fused_shared_experts,
        routed_scaling_factor, apply_routed_scaling_factor_on_output,
    )


def _install_function(function):
    """Retain both the compiled wrapper and aliases of its original callable."""
    with _LOCK:
        # Some Torch builds insert another functools-wrapped closure between
        # the compiled entry and the native body. Keep that wrapper too.
        original_callable = inspect.unwrap(function)
        if (not isinstance(original_callable, types.FunctionType)
                or original_callable.__closure__):
            raise RuntimeError("moe_topk_order: expected closure-free native router")
        if hasattr(original_callable, _MARKER):
            original, adapted, stock, ordered = getattr(original_callable, _MARKER)
            if (original_callable.__code__ is not _native.__code__
                    or original_callable.__globals__.get(_DISPATCH)
                    is not ordered
                    or original_callable.__globals__.get(_STOCK) is not stock
                    or stock.__code__ is not original
                    or ordered.__code__ is not adapted):
                raise RuntimeError("moe_topk_order: binding changed after installation")
            return function
        signature = inspect.signature(original_callable)
        if (tuple(signature.parameters) != _PARAMETERS
                or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                       for p in signature.parameters.values())):
            raise RuntimeError("moe_topk_order: native router signature changed")
        instructions = list(dis.get_instructions(original_callable))
        calls = sum(
            item.opname == "LOAD_GLOBAL" and item.argval == "torch"
            and following.opname in ("LOAD_ATTR", "LOAD_METHOD")
            and following.argval == "topk"
            for item, following in zip(instructions, instructions[1:])
        )
        if calls != 2 or not {
            "group_idx", "topk_ids", "topk_weights", "scores_for_choice",
        }.issubset(original_callable.__code__.co_varnames):
            raise RuntimeError("moe_topk_order: native grouped routing body changed")
        namespace = original_callable.__globals__
        if (namespace.get("torch") is None
                or _DISPATCH in namespace or _STOCK in namespace):
            raise RuntimeError("moe_topk_order: missing Torch or foreign dispatch binding")
        original = original_callable.__code__
        adapted = _ordered_code(original)
        stock = types.FunctionType(original, namespace, original_callable.__name__,
                                   original_callable.__defaults__)
        ordered = types.FunctionType(adapted, namespace, original_callable.__name__,
                                     original_callable.__defaults__)
        stock.__kwdefaults__ = ordered.__kwdefaults__ = original_callable.__kwdefaults__
        namespace[_STOCK] = stock
        namespace[_DISPATCH] = ordered
        original_callable.__signature__ = signature
        setattr(original_callable, _MARKER, (original, adapted, stock, ordered))
        original_callable.__code__ = _native.__code__
        return function


def install():
    from . import hcu_env

    if not hcu_env.enabled("HCU_MOE_STABLE_TOPK"):
        return
    import torch

    if torch.version.hip is None:
        return
    from sglang.srt.layers.moe import topk

    _install_function(topk.biased_grouped_topk_impl)
    _reg.register(
        "platform_detect", "stable-native-grouped-topk",
        target="sglang.srt.layers.moe.topk.biased_grouped_topk_impl",
        mechanism=_reg.MECH_REBIND,
        detail="two validated GPU sorted operands before normalization; CPU and aliases retained; no source writes",
    )
    LOG.warning("HCU adapter: native grouped TopK order stabilized at runtime")
