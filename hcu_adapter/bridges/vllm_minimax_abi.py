# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed compatibility for the old MiniMax fused-op native ABI."""
import hashlib
import inspect
from importlib import import_module

from hcu_adapter.infra import registry
from hcu_adapter.infra.kill_switch import op_disabled

_NAMES = ("qkv", "q_norm_weight", "k_norm_weight", "cos_sin_cache", "positions",
          "num_heads", "num_kv_heads", "rotary_dim", "eps", "index_q_norm_weight",
          "index_k_norm_weight", "num_index_heads", "slot_mapping", "index_slot_mapping",
          "kv_cache", "index_cache", "block_size", "q_out", "index_q_out",
          "kv_cache_dtype", "skip_index_branch", "q_fp8_out", "q_fp8_scale",
          "kv_k_scale", "kv_v_scale", "write_qkv")
_SOURCE_SHA256 = "e6acdd8d074c080de2589224f5e83c133d3d086ecf5e47c9183945528ded39a6"
# Release wheel g7364da ships the original 25-argument Python wrapper together
# with the original 25-argument native op. It needs no compatibility bridge.
_OLD_NAMES = _NAMES[:-1]
_OLD_SOURCE_SHA256 = "7d5940234fa7f2c376ab979887781dbf9171afb5517a9ec83d610aed47f6cc29"
_OLD_SCHEMA = ("_C::fused_minimax_m3_qknorm_rope_kv_insert(Tensor($0! -> ) qkv, Tensor q_norm_weight, Tensor k_norm_weight, Tensor cos_sin_cache, Tensor positions, int num_heads, int num_kv_heads, int rotary_dim, float eps, Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, int num_index_heads, Tensor? slot_mapping, Tensor? index_slot_mapping, Tensor($1! -> )? kv_cache, Tensor($2! -> )? index_cache, int block_size, Tensor($3! -> )? q_out, Tensor($4! -> )? index_q_out, str kv_cache_dtype, bool skip_index_branch=False, Tensor($5! -> )? q_fp8_out=None, float q_fp8_scale=1., Tensor? kv_k_scale=None, Tensor? kv_v_scale=None) -> ()")
_NEW_SCHEMA = _OLD_SCHEMA.replace("Tensor? kv_v_scale=None) -> ()", "Tensor? kv_v_scale=None, bool write_qkv=True) -> ()")
_MARKER = "_hcu_minimax_old_abi_call"


def _replacement(qkv, q_norm_weight, k_norm_weight, cos_sin_cache, positions,
                 num_heads, num_kv_heads, rotary_dim, eps, index_q_norm_weight,
                 index_k_norm_weight, num_index_heads, slot_mapping, index_slot_mapping,
                 kv_cache, index_cache, block_size, q_out, index_q_out, kv_cache_dtype,
                 skip_index_branch, q_fp8_out, q_fp8_scale, kv_k_scale, kv_v_scale,
                 write_qkv):
    return _hcu_minimax_old_abi_call(qkv, q_norm_weight, k_norm_weight, cos_sin_cache,
        positions, num_heads, num_kv_heads, rotary_dim, eps, index_q_norm_weight,
        index_k_norm_weight, num_index_heads, slot_mapping, index_slot_mapping,
        kv_cache, index_cache, block_size, q_out, index_q_out, kv_cache_dtype,
        skip_index_branch, q_fp8_out, q_fp8_scale, kv_k_scale, kv_v_scale, write_qkv)


def install():
    ops = import_module("vllm._custom_ops")
    torch = import_module("torch")
    fn = getattr(ops, "fused_minimax_m3_qknorm_rope_kv_insert", None)
    native = getattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert", None)
    if fn is None or native is None:
        return False
    schema = str(native.default._schema)
    if schema == _NEW_SCHEMA:
        # Even the new native ABI is safe only with a known matching Python
        # wrapper.  A schema match alone does not establish call semantics.
        if (tuple(inspect.signature(fn).parameters) != _NAMES
                or fn.__closure__ is not None
                or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != _SOURCE_SHA256
                or _MARKER in fn.__globals__
                or getattr(fn, "_hcu_minimax_abi_installed", False)):
            raise RuntimeError("MiniMax fused Python wrapper drift; refusing new native ABI")
        return False
    if schema != _OLD_SCHEMA:
        raise RuntimeError("MiniMax fused native schema drift; refusing compatibility bridge")
    if (tuple(inspect.signature(fn).parameters) == _OLD_NAMES
            and fn.__closure__ is None
            and hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() == _OLD_SOURCE_SHA256
            and _MARKER not in fn.__globals__):
        # Both sides are the known old ABI; changing the Python code object
        # would only add an unnecessary compatibility layer.
        return False
    if op_disabled("vllm_minimax_qknorm_abi"):
        raise RuntimeError("MiniMax old native ABI bridge disabled; unsupported schema")
    if getattr(fn, "_hcu_minimax_abi_installed", False):
        if fn.__globals__.get(_MARKER) is None or fn.__code__ is not _replacement.__code__:
            raise RuntimeError("MiniMax fused ABI bridge ownership changed")
        return False
    if (tuple(inspect.signature(fn).parameters) != _NAMES or fn.__closure__ is not None
            or hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != _SOURCE_SHA256
            or _MARKER in fn.__globals__):
        raise RuntimeError("MiniMax fused Python wrapper drift; refusing compatibility bridge")
    def call(*args):
        if args[-1] is not True:
            raise NotImplementedError("native MiniMax fused ABI requires write_qkv=True")
        return native(*args[:-1])
    registry.register(op="vllm_minimax_qknorm_abi", impl="native-25",
        target="vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert",
        mechanism=registry.MECH_REBIND, source=__name__,
        detail="25-argument native ABI; write_qkv=True only; preserves wrapper aliases")
    fn.__globals__[_MARKER] = call
    fn.__code__ = _replacement.__code__
    fn._hcu_minimax_abi_installed = True
    return True
