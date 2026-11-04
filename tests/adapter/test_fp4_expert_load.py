# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""FP4 loading: real community functions, metadata isolation and numeric refs."""

import ast
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from hcu_adapter import fp4_expert_load as bridge

_TABLE = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
_LOAD = """def process_weights_after_loading_block_quant(self, layer):
    if self.use_mxfp8:
        return self._process_mxfp8_moe_weights(layer, quantize=False)
    if self.is_fp4_expert and self.dequant_fp4_to_fp8:
        for weight, scale in [(layer.w13_weight, layer.w13_weight_scale_inv),
                              (layer.w2_weight, layer.w2_weight_scale_inv)]:
            ws, ss = [], []
            for e in range(weight.shape[0]):
                w, s = cast_e2m1fn_to_e4m3fn(weight.data[e], scale.data[e])
                ws.append(w)
                ss.append(s)
            weight.data = torch.stack(ws)
            scale.data = torch.stack(ss).float()
            scale.format_ue8m0 = False
        self.is_fp4_expert = False
    if not self.is_fp4_expert:
        for weight, scale in [(layer.w13_weight, layer.w13_weight_scale_inv),
                              (layer.w2_weight, layer.w2_weight_scale_inv)]:
            requant_block_scale_ue8m0_for_deepgemm(
                weight, scale, self.quant_config.weight_block_size,
                use_deepgemm_runner=False, output_dtype=torch.bfloat16,
                weight_shape=weight.shape[-2:])
"""
_HELPER = """def _dequantize_aiter_fp4_experts(self, layer):
    for weight, scale in [(layer.w13_weight, layer.w13_weight_scale_inv),
                          (layer.w2_weight, layer.w2_weight_scale_inv)]:
        pairs = [cast_e2m1fn_to_e4m3fn(weight.data[e], scale.data[e])
                 for e in range(weight.shape[0])]
        weight.data = torch.stack([w for w, s in pairs])
        scale.data = torch.stack([s for w, s in pairs]).float()
        scale.format_ue8m0 = False
    self.is_fp4_expert = False
"""


def _unpack(packed, scales):
    # Independent per-32 FP4 decoding; no block-FP8/5D reshape algorithm.
    table = torch.tensor(_TABLE, device=packed.device)
    raw = packed.to(torch.int16)
    codes = torch.stack([raw & 15, (raw >> 4) & 15], dim=-1)
    values = table[codes.long()].flatten(-2)
    return values * scales.repeat_interleave(32, dim=-1)


def _offline_cast(packed, scales):
    values = _unpack(packed, scales)
    return (values.to(torch.float8_e4m3fn),
            torch.ones(values.shape[0] // 128, values.shape[1] // 128,
                       device=values.device))


@pytest.fixture(params=["offline", "community-git"])
def source(request):
    if request.param == "offline":
        return _LOAD + "\n" + _HELPER
    tree = os.environ.get("HCU_TEST_SGLANG_TREE")
    if not tree:
        pytest.skip("requires explicit community Git tree")
    tree = Path(tree).resolve()
    text = subprocess.check_output([
        "git", "-c", f"safe.directory={tree}", "-C", str(tree), "show",
        "HEAD:python/sglang/srt/layers/quantization/fp8.py",
    ]).decode()
    nodes = ast.parse(text).body
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "Fp8MoEMethod")
    names = {"process_weights_after_loading_block_quant", "_dequantize_aiter_fp4_experts"}
    parts = [ast.get_source_segment(text, n) for n in cls.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    cast = next(n for n in nodes
                if isinstance(n, ast.FunctionDef) and n.name == bridge._CAST)
    table = next(n for n in nodes if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "DSV4_DEQUANT_FP4_TABLE"
                         for t in n.targets))
    return "\n\n".join([ast.get_source_segment(text, table),
                       ast.get_source_segment(text, cast), *parts])


def _method(source, monkeypatch, device="cpu", experts=1, rows=128, columns=128,
            filename="<fp4-community>"):
    calls = []
    backend = types.SimpleNamespace(is_megamoe=lambda: False,
                                    is_flashinfer_megamoe=lambda: False)
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    dg = types.ModuleType("sglang.srt.layers.deep_gemm_wrapper")
    dg.DEEPGEMM_SCALE_UE8M0 = False
    monkeypatch.setitem(sys.modules, dg.__name__, dg)
    monkeypatch.setattr(sys.modules["sglang.srt.layers"], "deep_gemm_wrapper", dg,
                        raising=False)
    ns = {
        "torch": torch, bridge._CAST: _offline_cast,
        "logger": types.SimpleNamespace(warning_once=lambda *a: None),
        "_use_aiter": False, "_is_fp8_fnuz": False, "_is_cpu": False,
        "_is_npu": False, "get_moe_a2a_backend": lambda: backend,
        "get_platform": lambda: types.SimpleNamespace(is_sm90=False),
        "requant_block_scale_ue8m0_for_deepgemm": lambda *a, **k: calls.append((a, k)),
    }
    exec(compile("from __future__ import annotations\n" + source, filename, "exec"), ns)
    cls = type("Fp8MoEMethod", (), {
        "process_weights_after_loading_block_quant": ns["process_weights_after_loading_block_quant"],
        "_dequantize_aiter_fp4_experts": ns["_dequantize_aiter_fp4_experts"],
        "is_deepgemm_moe_runner_backend_enabled": lambda self: False,
        "_process_mxfp8_moe_weights": lambda self, *a, **k: calls.append("mxfp8"),
    })
    method = cls()
    method.is_fp4_expert = True
    method.dequant_fp4_to_fp8 = True
    method.use_mxfp8 = False
    method.convert_mxfp8_to_block = False
    method.quant_config = types.SimpleNamespace(
        weight_block_size=[32, 32], is_checkpoint_fp8_serialized=False)
    method.weight_block_size = [32, 32]
    layer = torch.nn.Module()
    references = {}
    for name, n, k in (("w13", rows, columns), ("w2", columns, rows)):
        raw = (torch.arange(experts * n * k // 2).reshape(experts, n, k // 2)
               % 256).to(torch.int8).to(device)
        powers = torch.arange(experts * n * k // 32).reshape(experts, n, k // 32) % 5 - 2
        scales = (2.0 ** powers).to(device)
        references[name] = _unpack(raw, scales).to(torch.bfloat16)
        setattr(layer, name + "_weight", torch.nn.Parameter(raw, requires_grad=False))
        setattr(layer, name + "_weight_scale_inv",
                torch.nn.Parameter(scales, requires_grad=False))
    return method, layer, references, calls, ns


def test_aliases_readonly_source_and_block_metadata(source, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "0")
    path = tmp_path / "community_fp8.py"
    path.write_text(source, encoding="utf-8")
    path.chmod(0o444)
    before = hashlib.sha256(path.read_bytes()).digest()
    try:
        method, layer, refs, calls, ns = _method(source, monkeypatch, filename=str(path))
        function = type(method).process_weights_after_loading_block_quant
        alias, signature, code = method.process_weights_after_loading_block_quant, inspect.signature(function), function.__code__
        shared, cast = method.quant_config, ns[bridge._CAST]
        assert bridge._install_method(type(method)) is function
        assert bridge._install_method(type(method)) is function
        assert getattr(function, bridge._MARKER)[0].__code__ is code
        alias(layer=layer)
        assert inspect.signature(function) == signature
        assert ns[bridge._CAST] is cast  # no module-wide cast replacement
        assert shared.weight_block_size == [32, 32]
        assert method.quant_config is not shared
        assert method.quant_config.weight_block_size == method.weight_block_size == [128, 128]
        assert len(calls) == 2 and all(args[2] == [128, 128] for args, _ in calls)
        for name in refs:
            weight = getattr(layer, name + "_weight")
            scale = getattr(layer, name + "_weight_scale_inv")
            expanded = scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
            torch.testing.assert_close((weight.float() * expanded).bfloat16(), refs[name],
                                       rtol=0, atol=0)
        assert not method.is_fp4_expert
        assert hashlib.sha256(path.read_bytes()).digest() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("experts,rows,columns", [(1, 128, 128), (2, 256, 384)])
def test_explicit_bf16_matches_independent_fp4_reference(source, monkeypatch,
                                                       experts, rows, columns):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "1")
    method, layer, refs, calls, _ = _method(
        source, monkeypatch, experts=experts, rows=rows, columns=columns)
    shared = method.quant_config
    bridge._install_method(type(method))
    method.process_weights_after_loading_block_quant(layer)
    assert not calls and not method.is_fp4_expert
    assert shared.weight_block_size == [32, 32]
    assert method.quant_config.weight_block_size == method.weight_block_size == [128, 128]
    for name, ref in refs.items():
        weight = getattr(layer, name + "_weight")
        assert weight.dtype == torch.bfloat16 and not weight.requires_grad
        assert getattr(layer, name + "_weight_scale_inv") is None
        torch.testing.assert_close(weight, ref, rtol=0, atol=0)


@pytest.mark.parametrize("route", ["fp8", "mxfp8"])
def test_nonconversion_routes_preserve_config(source, monkeypatch, route):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "0")
    method, layer, _, calls, _ = _method(source, monkeypatch)
    method.is_fp4_expert = False
    method.use_mxfp8 = route == "mxfp8"
    shared = method.quant_config
    bridge._install_method(type(method))
    method.process_weights_after_loading_block_quant(layer)
    assert method.quant_config is shared and method.weight_block_size == [32, 32]
    if route == "mxfp8":
        assert calls == ["mxfp8"]
    else:
        assert len(calls) == 2 and all(args[2] == [32, 32] for args, _ in calls)


def test_bad_cast_recipe_rejected_without_config_change(source, monkeypatch):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "0")
    method, layer, _, _, ns = _method(source, monkeypatch)
    ns[bridge._CAST] = lambda *a: (torch.zeros(128, 128), torch.ones(4, 4))
    bridge._install_method(type(method))
    shared, weight = method.quant_config, layer.w13_weight
    with pytest.raises(RuntimeError, match="block recipe changed"):
        method.process_weights_after_loading_block_quant(layer)
    assert method.quant_config is shared and layer.w13_weight is weight


def test_second_bf16_failure_does_not_partially_replace_parameters(source, monkeypatch):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "1")
    method, layer, _, _, _ = _method(source, monkeypatch)
    bridge._install_method(type(method))
    original = bridge._block_to_bf16
    calls = []

    def fail_second(w, s):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second conversion failure")
        return original(w, s)

    monkeypatch.setattr(bridge, "_block_to_bf16", fail_second)
    with pytest.raises(RuntimeError, match="second conversion failure"):
        method.process_weights_after_loading_block_quant(layer)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.float8_e4m3fn
    assert layer.w13_weight_scale_inv is not None and layer.w2_weight_scale_inv is not None
    assert method.weight_block_size == [32, 32]


def test_foreign_and_drift_bindings_rejected(source, monkeypatch):
    method, _, _, _, ns = _method(source, monkeypatch)
    function = type(method).process_weights_after_loading_block_quant
    before = function.__code__
    ns[bridge._TARGET] = object()
    with pytest.raises(RuntimeError, match="foreign"):
        bridge._install_method(type(method))
    assert function.__code__ is before
    del ns[bridge._TARGET]
    bridge._install_method(type(method))
    ns[bridge._TARGET] = lambda *a: None
    with pytest.raises(RuntimeError, match="changed after installation"):
        bridge._install_method(type(method))


def test_public_install_and_compatibility_entries_keep_alias(source, monkeypatch, tmp_path):
    from hcu_adapter import dsv4_ep_bf16, dsv4_fp4_dequant_blockshape, registry

    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "0")
    method, layer, _, calls, _ = _method(source, monkeypatch)
    alias = type(method).process_weights_after_loading_block_quant
    for name in ("sglang.srt.layers.quantization",):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = types.ModuleType("sglang.srt.layers.quantization.fp8")
    module.__file__ = str(tmp_path / "missing-fp8.py")
    module.Fp8MoEMethod = type(method)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    records = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: records.append(k))
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "torch",
                        types.SimpleNamespace(version=types.SimpleNamespace(hip="6.3")))
        bridge.install()
        dsv4_ep_bf16.install()
        dsv4_fp4_dequant_blockshape.install()
    alias(self=method, layer=layer)
    assert len(calls) == 2
    assert len(records) == 3 and all(r["mechanism"] == registry.MECH_REBIND for r in records)
    assert not list(tmp_path.iterdir())


def _spawn_worker(source, queue):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("SGLANG_HCU_EP_BF16", "1")
        method, layer, refs, _, _ = _method(source, monkeypatch)
        alias = method.process_weights_after_loading_block_quant
        bridge._install_method(type(method))
        alias(layer)
        queue.put(all(torch.equal(getattr(layer, n + "_weight"), ref)
                      for n, ref in refs.items()))


def test_spawned_loader_retains_community_math(source):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_worker, args=(source, queue))
    process.start()
    try:
        assert queue.get(timeout=40) is True
        process.join(timeout=20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        queue.close()


def test_shared_checkpoint_config_isolated_between_methods(source, monkeypatch):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "0")
    method, layer, _, _, _ = _method(source, monkeypatch)
    other = type(method)()
    other.quant_config = method.quant_config
    other.weight_block_size = [32, 32]
    bridge._install_method(type(method))
    method.process_weights_after_loading_block_quant(layer)
    assert other.quant_config.weight_block_size == other.weight_block_size == [32, 32]
    assert method.quant_config is not other.quant_config


@pytest.mark.parametrize("change", ["signature", "closure", "missing-cast", "missing-helper"])
def test_loader_contract_drift_rejected_before_mutation(monkeypatch, change):
    method, _, _, _, _ = _method(_LOAD + "\n" + _HELPER, monkeypatch)
    cls = type(method)
    if change == "signature":
        cls.process_weights_after_loading_block_quant = lambda self, layer, extra=1: None
    elif change == "closure":
        value = object()

        def closure(self, layer):
            return value

        cls.process_weights_after_loading_block_quant = closure
    elif change == "missing-cast":
        cls.process_weights_after_loading_block_quant = lambda self, layer: None
    else:
        cls._dequantize_aiter_fp4_experts = None
    before = cls.process_weights_after_loading_block_quant.__code__
    with pytest.raises(RuntimeError):
        bridge._install_method(cls)
    assert cls.process_weights_after_loading_block_quant.__code__ is before


def test_cuda_install_is_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(version=types.SimpleNamespace(hip=None)))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.fp8", None)
    bridge.install()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires remote GPU")
@pytest.mark.parametrize("bf16", [False, True])
def test_gpu_original_cast_and_loader(source, monkeypatch, bf16):
    monkeypatch.setenv("SGLANG_HCU_EP_BF16", "1" if bf16 else "0")
    method, layer, refs, calls, _ = _method(
        source, monkeypatch, device="cuda", experts=2, rows=256, columns=384)
    bridge._install_method(type(method))
    method.process_weights_after_loading_block_quant(layer)
    for name, ref in refs.items():
        w = getattr(layer, name + "_weight")
        if bf16:
            actual = w
            assert getattr(layer, name + "_weight_scale_inv") is None
        else:
            s = getattr(layer, name + "_weight_scale_inv")
            actual = (w.float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)).bfloat16()
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    assert method.weight_block_size == [128, 128]
