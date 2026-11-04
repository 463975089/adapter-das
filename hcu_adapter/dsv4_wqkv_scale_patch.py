# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU adapter: fuse the CT int8 per-channel scales into wqkv_a (DSV4).

v0522 ``deepseek_v4.py`` load_weights fuses checkpoint ``wq_a``/``wkv``
pairs into the model's fused ``wqkv_a`` param (SGLANG_OPT_FUSE_WQA_WKV on
for the DSV4.1 family). The fuse branch catches ``.weight`` /
``.weight_scale_inv`` (fp8 naming) / ``.qweight`` (GGUF) — but NOT
``.weight_scale``, the compressed-tensors **INT8 per-channel scale**
name. DAS w4a8 checkpoints (DeepSeek-V4.1-Flash-Channel-INT4-w4a8) carry
exactly that name: the fused ``wqkv_a.weight`` gets its data but the q/kv
scales fall through to "not found in params_dict" and are dropped, so
``wqkv_a.weight_scale`` (created by CompressedTensorsW8A8Int8
.create_weights, verified v0522 compressed_tensors_w8a8_int8.py:78) never
receives data — every attention projection dequantizes with an
uninitialized scale. Boot-fine, numerics-garbage; nmz16 2026-10-26
(warnings during shard load; scheme/target forensics same day: the
WeigthsMapper rewrites the group targets to ``self_attn.(?:wqkv_a|wq_b)``
so the unfused names have no scheme either — disabling the fuse does NOT
dodge this, it makes wq_a/wkv fully unquantized instead).

Platform-independent upstream gap. The runtime bridge extends only the two
existing str.endswith arguments to tuples accepted by that same API. The
original bytecode, lazy weight iterator, filtering, fusion and parameter
loading stay in use. No source reads, recompilation or file writes.
Gate: HCU_DSV4_WQKV_SCALE=0 skips installation.
"""

import dis
import inspect
import logging
import threading
import types

LOG = logging.getLogger("hcu.adapter")
_LOCK = threading.RLock()
_MARKER = "_hcu_ct_wqkv_scale_codes"
_SUFFIXES = {
    ".wq_a.weight_scale_inv": ".wq_a.weight_scale",
    ".wkv.weight_scale_inv": ".wkv.weight_scale",
}


def _check_suffix(code, suffix):
    instructions = list(dis.get_instructions(code))
    sites = [i for i, op in enumerate(instructions)
             if op.opname == "LOAD_CONST" and op.argval == suffix]
    if len(sites) != 1:
        raise RuntimeError("dsv4_wqkv_scale: upstream suffix site changed")
    i = sites[0]
    if (i < 2 or instructions[i - 1].opname not in ("LOAD_METHOD", "LOAD_ATTR")
            or instructions[i - 1].argval != "endswith"
            or instructions[i - 2].opname != "LOAD_FAST"
            or instructions[i - 2].argval != "name"):
        raise RuntimeError("dsv4_wqkv_scale: suffix is no longer a name.endswith argument")
    call = instructions[i + 1]
    if call.opname not in ("CALL_METHOD", "PRECALL", "CALL") or call.arg != 1:
        raise RuntimeError("dsv4_wqkv_scale: upstream endswith call arguments changed")


def _install_function(function):
    with _LOCK:
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise RuntimeError("dsv4_wqkv_scale: expected closure-free load_weights")
        if hasattr(function, _MARKER):
            original, adapted = getattr(function, _MARKER)
            if function.__code__ is not adapted:
                raise RuntimeError("dsv4_wqkv_scale: loader changed after installation")
            return function
        parameters = inspect.signature(function).parameters
        if (list(parameters) != ["self", "weights", "is_nextn"]
                or parameters["is_nextn"].default is not False
                or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                       for p in parameters.values())
                or any(parameters[n].default is not inspect.Parameter.empty
                       for n in ("self", "weights"))):
            raise RuntimeError("dsv4_wqkv_scale: upstream loader signature changed")
        original = function.__code__
        replacements = {}
        for old, extra in _SUFFIXES.items():
            _check_suffix(original, old)
            if extra in original.co_consts:
                _check_suffix(original, extra)  # community may already support it
            else:
                replacements[old] = (old, extra)
        adapted = original.replace(co_consts=tuple(
            replacements.get(value, value) if isinstance(value, str) else value
            for value in original.co_consts
        )) if replacements else original
        setattr(function, _MARKER, (original, adapted))
        function.__code__ = adapted
        return function


def install():
    from . import hcu_env, registry

    if not hcu_env.enabled("HCU_DSV4_WQKV_SCALE"):
        return
    import sglang.srt.models.deepseek_v4 as model

    _install_function(model.DeepseekV4ForCausalLM.load_weights)
    registry.register(
        "platform_detect", "dsv4-wqkv-scale-fuse",
        target="sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
        mechanism=registry.MECH_REBIND,
        detail="extend two str.endswith argument tuples; original loader and aliases retained; no source writes",
    )
    LOG.warning("HCU adapter: DSV4 CT-int8 wqkv scale loading adapted at runtime")
