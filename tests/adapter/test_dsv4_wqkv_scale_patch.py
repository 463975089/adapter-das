"""CT scale fusion through the original loader, without source modification."""

import ast
import concurrent.futures
import hashlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest
import torch

from hcu_adapter import dsv4_wqkv_scale_patch as bridge

_SOURCE = """def load_weights(self, weights, is_nextn=False):
    params = dict(self.named_parameters())
    cache = {}
    names = []
    for name, weight in weights:
        name = self.remap_weight_name_to_dpsk_hf_format(
            name, is_nextn=is_nextn, num_hidden_layers=self.config.num_hidden_layers)
        layer = get_layer_id(name)
        if layer is not None and (layer < self.model.start_layer or layer >= self.model.end_layer):
            continue
        if is_nextn:
            prefix = "model.layers." + str(self.config.num_hidden_layers)
            if not name.startswith(prefix):
                continue
            name = name.replace(prefix, "model.decoder")
        elif name.startswith("mtp"):
            continue
        names.append(name)
        if (envs.SGLANG_OPT_FUSE_WQA_WKV.get()
                and ".compressor." not in name and ".engram." not in name
                and (name.endswith(".wq_a.weight") or name.endswith(".wkv.weight")
                     or name.endswith(".wq_a.weight_scale_inv")
                     or name.endswith(".wkv.weight_scale_inv"))):
            is_q = ".wq_a." in name
            target = name.replace(".wq_a." if is_q else ".wkv.", ".wqkv_a.")
            bucket = cache.setdefault(target, {})
            shard = "q" if is_q else "kv"
            assert shard not in bucket, "duplicate shard"
            bucket[shard] = _clone_if_runai_streamed_tensor(weight)
            if len(bucket) == 2:
                params[target].copy_(_fuse_deepseek_v4_wqkv_a_pair(target, bucket))
                cache.pop(target)
        elif name in params:
            params[name].copy_(weight)
    assert not cache, "unpaired shard"
    self.post_load_weights(is_nextn=is_nextn, weight_names=names)
"""


@pytest.fixture(params=["offline", "community-git"])
def source(request):
    if request.param == "offline":
        return _SOURCE
    tree = os.environ.get("HCU_TEST_SGLANG_TREE")
    if not tree:
        pytest.skip("requires explicitly selected community Git tree")
    tree = Path(tree).resolve()
    text = subprocess.check_output([
        "git", "-c", f"safe.directory={tree}", "-C", str(tree), "show",
        "HEAD:python/sglang/srt/models/deepseek_v4.py",
    ]).decode()
    nodes = ast.parse(text).body
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ForCausalLM")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_weights")
    helper = next(n for n in nodes if isinstance(n, ast.FunctionDef)
                  and n.name == "_fuse_deepseek_v4_wqkv_a_pair")
    return ast.get_source_segment(text, helper) + "\n\n" + ast.get_source_segment(text, method)


def _model(source, *, fuse=True, nextn=False, device="cpu", filename="<community-dsv4>"):
    state, notes = {"fuse": fuse}, []
    tensors = {}
    prefix = "model.decoder" if nextn else "model.layers.0"
    if fuse:
        for suffix, shape in (("weight_scale", (5, 1)), ("weight", (5, 4))):
            dtype = torch.int8 if suffix == "weight" else torch.float32
            tensors[prefix + ".self_attn.wqkv_a." + suffix] = torch.full(
                shape, -99, dtype=dtype, device=device)
    else:
        for proj, rows in (("wq_a", 2), ("wkv", 3)):
            tensors[prefix + ".self_attn." + proj + ".weight_scale"] = torch.full(
                (rows, 1), -99., device=device)

    def submit(executor, futures, use_async, func, func_args, func_kwargs=None):
        if use_async:
            futures.append(executor.submit(func, *func_args, **(func_kwargs or {})))
        else:
            func(*func_args, **(func_kwargs or {}))

    ns = {
        "torch": torch, "concurrent": concurrent,
        "DEEPSEEK_V4_STACKED_PARAMS_MAPPING": [],
        "FusedMoE": types.SimpleNamespace(make_expert_params_mapping=lambda **k: []),
        "is_wint4afp8_or_wint4a16_config": lambda c: False,
        "envs": types.SimpleNamespace(SGLANG_OPT_FUSE_WQA_WKV=types.SimpleNamespace(
            get=lambda: state["fuse"])),
        "_clone_if_runai_streamed_tensor": lambda w: w.clone(),
        "_fuse_deepseek_v4_wqkv_a_pair": lambda name, bucket: torch.cat([bucket["q"], bucket["kv"]]),
        "should_async_load": lambda w: state.get("async", False),
        "maybe_executor_submit": submit,
        "default_weight_loader": lambda param, weight: param.copy_(weight),
        "get_layer_id": lambda name: (
            int(re.search(r"model\.layers\.(\d+)", name).group(1))
            if re.search(r"model\.layers\.(\d+)", name) else None),
        "logger": types.SimpleNamespace(warning=lambda *a: notes.append(a)),
        "log_info_on_rank0": lambda *a: None,
    }
    exec(compile("from __future__ import annotations\n" + source, filename, "exec"), ns)
    cls = type("DeepseekV4ForCausalLM", (), {"load_weights": ns["load_weights"]})
    model = cls()
    model.config = types.SimpleNamespace(num_hidden_layers=2, n_routed_experts=0,
                                        model_type="deepseek_v4")
    if nextn:
        model.config.num_nextn_predict_layers = 1
    model.wo_a_fp8 = True
    model.quant_config = None
    model.num_fused_shared_experts = 0
    model.model = types.SimpleNamespace(start_layer=0, end_layer=3 if nextn else 2, layers=[])
    model.named_parameters = lambda: tensors.items()
    model.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.remap_weight_name_to_dpsk_hf_format = lambda name, **k: name
    model.post_load_weights = lambda **k: notes.append(k)
    model._prewarm_mhc_kernels = lambda: None
    return model, tensors, ns, state, notes


def _weights(device="cpu", nextn=False):
    prefix = "model.layers.2" if nextn else "model.layers.0"
    q = torch.tensor([[.25], [2.]], device=device)
    kv = torch.tensor([[.5], [3.], [1.25]], device=device)
    qw = torch.tensor([[2, -3, 5, 1], [-2, 1, 4, -5]], dtype=torch.int8, device=device)
    kw = torch.tensor([[3, -1, 2, 4], [2, 2, -4, 3], [-5, 1, 2, -3]],
                      dtype=torch.int8, device=device)
    return [
        (prefix + ".self_attn.wkv.weight_scale", kv),
        (prefix + ".self_attn.wq_a.weight", qw),
        (prefix + ".self_attn.wq_a.weight_scale", q),
        (prefix + ".self_attn.wkv.weight", kw),
    ], (q, kv, qw, kw)


def _assert_loaded(tensors, data, device="cpu", nextn=False):
    prefix = "model.decoder" if nextn else "model.layers.0"
    q, kv, qw, kw = data
    scale = tensors[prefix + ".self_attn.wqkv_a.weight_scale"]
    weight = tensors[prefix + ".self_attn.wqkv_a.weight"]
    assert torch.equal(scale, torch.cat([q, kv]))
    assert torch.equal(weight, torch.cat([qw, kw]))
    x = torch.tensor([[1., -2., .5, 3.], [-1., 0., 4., .25]], device=device)
    actual = x @ (weight.float() * scale).T
    expected = torch.cat([x @ (qw.float() * q).T, x @ (kw.float() * kv).T], dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_real_loader_alias_bytecode_and_readonly_source(source, tmp_path):
    path = tmp_path / "deepseek_v4.py"
    path.write_text(source, encoding="utf-8")
    path.chmod(0o444)
    before = hashlib.sha256(path.read_bytes()).digest()
    try:
        model, tensors, ns, _, _ = _model(source, filename=str(path))
        function = type(model).load_weights
        alias, signature, original = model.load_weights, inspect.signature(function), function.__code__
        globals_before = dict(ns)
        assert bridge._install_function(function) is function
        assert bridge._install_function(function) is function
        assert function.__code__.co_code == original.co_code
        assert function.__code__.co_names == original.co_names
        assert inspect.signature(function) == signature
        assert ns == globals_before
        weights, data = _weights()
        alias(weights=iter(weights))
        _assert_loaded(tensors, data)
        assert hashlib.sha256(path.read_bytes()).digest() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("nextn,async_load", [(False, False), (False, True), (True, False)])
def test_original_filtering_nextn_and_async_paths(source, nextn, async_load):
    model, tensors, _, state, notes = _model(source, nextn=nextn)
    state["async"] = async_load
    bridge._install_function(type(model).load_weights)
    weights, data = _weights(nextn=nextn)
    # An unmatched scale in an excluded pipeline layer must still be ignored.
    weights.insert(0, ("model.layers.50.self_attn.wq_a.weight_scale", torch.ones(2, 1)))
    model.load_weights(iter(weights), is_nextn=nextn)
    _assert_loaded(tensors, data, nextn=nextn)
    assert any(isinstance(n, dict) and n["is_nextn"] == nextn for n in notes)


def test_unfused_scales_remain_separate(source):
    model, tensors, _, _, _ = _model(source, fuse=False)
    bridge._install_function(type(model).load_weights)
    weights, data = _weights()
    model.load_weights((n, w) for n, w in weights if n.endswith("weight_scale"))
    for proj, expected in (("wq_a", data[0]), ("wkv", data[1])):
        assert torch.equal(tensors["model.layers.0.self_attn." + proj + ".weight_scale"],
                           expected)


@pytest.mark.parametrize("group", ["compressor", "engram"])
def test_original_exclusions_not_fused(source, group):
    model, tensors, _, _, _ = _model(source)
    weights, _ = _weights()
    excluded = []
    for name, tensor in weights:
        if name.endswith("weight_scale"):
            name = name.replace(".self_attn.", ".self_attn." + group + ".")
            tensors[name] = torch.full_like(tensor, -99)
            excluded.append((name, tensor))
    bridge._install_function(type(model).load_weights)
    model.load_weights(excluded)
    for name, expected in excluded:
        assert torch.equal(tensors[name], expected)
    assert torch.all(tensors["model.layers.0.self_attn.wqkv_a.weight_scale"] == -99)


def test_orphan_scales_fail_in_original_fusion_logic(source):
    model, _, _, _, _ = _model(source)
    bridge._install_function(type(model).load_weights)
    weights, _ = _weights()
    with pytest.raises(AssertionError):
        model.load_weights([weights[0]])


def test_duplicate_scale_shards_fail_in_original_fusion_logic(source):
    model, _, _, _, _ = _model(source)
    bridge._install_function(type(model).load_weights)
    weights, _ = _weights()
    # Upstream Python 3.10's exception.add_note error may mask the assertion;
    # retain and inspect its context instead of replacing upstream diagnostics.
    with pytest.raises((AssertionError, AttributeError)) as exc:
        model.load_weights([weights[0], weights[0]])
    error = exc.value
    if isinstance(error, AttributeError):
        error = error.__context__
    assert isinstance(error, AssertionError) and "duplicate shard" in str(error)


def test_native_upstream_scale_support_is_idempotent():
    source = _SOURCE.replace(
        'or name.endswith(".wq_a.weight_scale_inv")',
        'or name.endswith(".wq_a.weight_scale_inv") or name.endswith(".wq_a.weight_scale")',
    ).replace(
        'or name.endswith(".wkv.weight_scale_inv")',
        'or name.endswith(".wkv.weight_scale_inv") or name.endswith(".wkv.weight_scale")',
    )
    model, tensors, _, _, _ = _model(source)
    original = type(model).load_weights.__code__
    bridge._install_function(type(model).load_weights)
    assert type(model).load_weights.__code__ is original
    weights, data = _weights()
    model.load_weights(weights)
    _assert_loaded(tensors, data)


@pytest.mark.parametrize("source", [
    "def load_weights(self, weights, is_nextn=False): return weights",
    "def load_weights(self, weights, is_nextn=False):\n"
    '    a = ".wq_a.weight_scale_inv"\n    return ".wkv.weight_scale_inv"\n',
    "def load_weights(self, weights): return weights",
])
def test_drift_rejected_before_mutation(source):
    ns = {}
    exec(source, ns)
    function = ns["load_weights"]
    original = function.__code__
    with pytest.raises(RuntimeError):
        bridge._install_function(function)
    assert function.__code__ is original


def test_post_install_code_changes_are_rejected():
    model, _, _, _, _ = _model(_SOURCE)
    function = type(model).load_weights
    bridge._install_function(function)
    function.__code__ = function.__code__.replace(co_name="foreign")
    with pytest.raises(RuntimeError, match="changed after installation"):
        bridge._install_function(function)


def test_public_install_and_gate_without_source_file(source, monkeypatch, tmp_path):
    from hcu_adapter import hcu_env, registry

    model, tensors, _, _, _ = _model(source)
    module = types.ModuleType("sglang.srt.models.deepseek_v4")
    module.__file__ = str(tmp_path / "absent.py")
    module.DeepseekV4ForCausalLM = type(model)
    for name in ("sglang", "sglang.srt", "sglang.srt.models"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    records = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: records.append(k))
    monkeypatch.setattr(hcu_env, "enabled", lambda name: False)
    original = type(model).load_weights.__code__
    bridge.install()
    assert type(model).load_weights.__code__ is original
    monkeypatch.setattr(hcu_env, "enabled", lambda name: True)
    bridge.install()
    bridge.install()
    weights, data = _weights()
    model.load_weights(weights)
    _assert_loaded(tensors, data)
    assert len(records) == 2 and all(r["mechanism"] == registry.MECH_REBIND for r in records)
    assert not list(tmp_path.iterdir())


def _spawn_worker(source, queue):
    model, tensors, _, _, _ = _model(source)
    alias = model.load_weights
    bridge._install_function(type(model).load_weights)
    weights, data = _weights()
    alias(iter(weights))
    _assert_loaded(tensors, data)
    queue.put(True)


def test_spawned_interpreter_adapts_its_own_original_loader(source):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires remote GPU")
def test_gpu_scale_loading_and_dequantized_projection(source):
    model, tensors, _, _, _ = _model(source, device="cuda")
    bridge._install_function(type(model).load_weights)
    weights, data = _weights("cuda")
    model.load_weights(iter(weights))
    _assert_loaded(tensors, data, "cuda")
