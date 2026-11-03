"""FP8 storage views preserve factory caches, public dtype selection and aliases."""

import ast
import functools
import inspect
import multiprocessing
import os
from pathlib import Path
import subprocess
import types

import pytest
import torch

from hcu_adapter import hcu_env, jit_fp8_ffi as bridge

_CACHE = """def cache_once(fn):
    result_map = {}
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]
    return wrapper
"""
_FACTORY = """@cache_once
def _jit_module(in_dtype, out_dtype, group_size, scale_ue8m0, row_major, aligned,
                fuse_silu_and_mul, masked_layout, use_pdl, sorted_layout=False):
    assert in_dtype in _SUPPORTED_INPUT_DTYPES
    assert out_dtype in _SUPPORTED_OUTPUT_DTYPES
    return load_jit("quant", in_dtype, out_dtype, group_size)
"""


@pytest.mark.parametrize("layout", ["contig", "masked"])
@pytest.mark.parametrize("group_size", [32, 128])
def test_gpu_dsv4_output_storage_and_variable_graph_replay(layout, group_size):
    if os.environ.get("HCU_TEST_JIT_GPU") != "1":
        pytest.skip("requires explicit remote HIP compiler validation")
    assert torch.version.hip and torch.cuda.is_available()
    from hcu_adapter import jit_sources
    from sglang.kernels.ops.moe import dsv4

    jit_sources.install()
    bridge.install()
    shape = (2, 5, 1024) if layout == "masked" else (5, 1024)
    x = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros((*shape[:-1], 512), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((*shape[:-1], 512 // group_size), device="cuda")
    counts = torch.tensor([3, 5], device="cuda", dtype=torch.int32)

    def run():
        if layout == "masked":
            dsv4.silu_and_mul_masked_post_quant(
                x, q, scales, group_size, counts, topk=2)
        else:
            dsv4.silu_and_mul_contig_post_quant(x, q, scales, group_size)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    generator = torch.Generator(device="cuda").manual_seed(92)
    for magnitude, lengths in zip((.25, 2., 7.), ((3, 5), (5, 2), (1, 4))):
        x.copy_(torch.randn(shape, device="cuda", generator=generator) * magnitude)
        counts.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        gate, up = x.float().chunk(2, -1)
        activation = torch.nn.functional.silu(gate) * up
        groups = activation.reshape(*shape[:-1], -1, group_size)
        ref_s = groups.abs().amax(-1).clamp_min(1e-10) / 448.
        ref_q = (groups / ref_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        # Masked rows outside each expert's count have no output contract.
        valid = (torch.arange(5, device="cuda")[None, :] < counts[:, None]
                 if layout == "masked" else slice(None))
        torch.testing.assert_close(scales[valid], ref_s[valid], rtol=2e-5, atol=1e-7)
        actual = (q.float().reshape_as(groups) * scales[..., None])[valid].flatten()
        want = (ref_q.float() * ref_s[..., None])[valid].flatten()
        torch.testing.assert_close(actual, want, rtol=.13, atol=magnitude ** 2 * .02)
        assert torch.nn.functional.cosine_similarity(actual, want, dim=0) >= .9999
        assert torch.nn.functional.cosine_similarity(
            actual, activation[valid].flatten(), dim=0) >= .999


@pytest.fixture(params=["offline", "community-git"])
def sources(request):
    if request.param == "offline":
        return _CACHE, _FACTORY
    tree = os.environ.get("HCU_TEST_SGLANG_TREE")
    if not tree:
        pytest.skip("requires explicitly selected community Git tree")

    def extract(path, name):
        text = subprocess.check_output([
            "git", "-c", f"safe.directory={tree}", "-C", tree, "show", "HEAD:" + path,
        ]).decode()
        node = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(text, node)

    prefix = "python/sglang/kernels/"
    return (
        extract(prefix + "jit/utils/common.py", "cache_once"),
        "@cache_once\n" + extract(prefix + "ops/quantization/per_token_group_quant.py", "_jit_module"),
    )


def _factory(sources, entry="per_token_group_quant"):
    modules, calls = [], []

    def native(x, output, scales, *args, **kwargs):
        calls.append((x, output, scales))
        if output.dtype == torch.float8_e4m3fn:
            raise RuntimeError("native FFI expects uint8 storage")
        if output.dtype == torch.uint8:
            output.fill_(56)  # e4m3fn bytes for +1, not a quantization reference

    def load(*args, **kwargs):
        calls.append(("load", args))
        module = types.SimpleNamespace(**{entry: native}, untouched=object())
        modules.append(module)
        return module

    ns = {
        "functools": functools, "torch": torch, "load_jit": load,
        "make_cpp_args": lambda *args: "args",
        "_SUPPORTED_INPUT_DTYPES": (torch.bfloat16, torch.float16, torch.float32),
        "_SUPPORTED_OUTPUT_DTYPES": (torch.float8_e4m3fn, torch.int8),
        "_SUPPORTED_GROUP_SIZES": (16, 32, 64, 128, 256),
    }
    exec("from __future__ import annotations\n" + "\n\n".join(sources), ns)
    return ns["_jit_module"], modules, calls


def _key(dtype=torch.float8_e4m3fn):
    return torch.bfloat16, dtype, 32, False, True, True, False, False, False


def test_original_factory_and_warm_aliases_use_only_output_view(sources):
    factory, modules, calls = _factory(sources)
    alias = factory
    signature = inspect.signature(factory)
    cached = alias(*_key())
    x = torch.ones(3, 4, dtype=torch.bfloat16)
    output = torch.empty(4, 3, dtype=torch.float8_e4m3fn).t()
    scale = torch.ones(3, 1)
    with pytest.raises(RuntimeError, match="expects uint8"):
        cached.per_token_group_quant(x, output, scale)
    bridge._install_factory(factory, "per_token_group_quant")
    proxy = alias(*_key())
    proxy.per_token_group_quant(x, output, scale)
    seen = calls[-1]
    assert seen[0] is x and seen[2] is scale
    assert seen[1].dtype == torch.uint8
    assert seen[1].data_ptr() == output.data_ptr()
    assert seen[1].stride() == output.stride()
    assert torch.equal(output.float(), torch.ones_like(output, dtype=torch.float32))
    assert len(modules) == 1 and factory is alias
    assert inspect.signature(factory) == signature
    assert proxy.untouched is cached.untouched
    bridge._install_factory(factory, "per_token_group_quant")
    assert alias(*_key()) is proxy
    # A previously unseen cache key is adapted too, without losing nominal
    # FP8 dtype selection inside the original factory.
    cold = list(_key())
    cold[2] = 128
    assert isinstance(alias(*cold), bridge._OutputStorage)
    assert len(modules) == 2


def test_other_dtypes_inputs_and_keyword_output_unchanged(sources):
    factory, _, calls = _factory(sources)
    bridge._install_factory(factory, "per_token_group_quant")
    module = factory(*_key())
    x = torch.ones((2, 4), dtype=torch.float8_e5m2)
    scale = torch.ones(2, 1)
    for dtype in (torch.int8, torch.uint8, torch.float8_e4m3fnuz, torch.float8_e5m2):
        output = torch.empty(2, 4, dtype=dtype)
        module.per_token_group_quant(x, output, scale)
        assert calls[-1][0] is x and calls[-1][1] is output and calls[-1][2] is scale
    output = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    module.per_token_group_quant(x=x, output=output, scales=scale)
    assert calls[-1][1].data_ptr() == output.data_ptr()
    assert calls[-1][1].dtype == torch.uint8


def test_drift_and_bad_cached_export_refused_without_partial_install(sources):
    factory, _, _ = _factory(sources)
    alias = factory
    cells = dict(zip(factory.__code__.co_freevars, factory.__closure__))
    cache = cells["result_map"].cell_contents
    old_fn = cells["fn"].cell_contents
    cache["bad"] = object()
    with pytest.raises(AttributeError):
        bridge._install_factory(factory, "per_token_group_quant")
    assert cells["fn"].cell_contents is old_fn
    assert cells["result_map"].cell_contents is cache
    assert not hasattr(factory, bridge._MARKER) and alias is factory
    cache.clear()
    bridge._install_factory(factory, "per_token_group_quant")
    cells["result_map"].cell_contents = {}
    with pytest.raises(RuntimeError, match="changed after"):
        bridge._install_factory(factory, "per_token_group_quant")
    with pytest.raises(RuntimeError, match="layout changed"):
        bridge._install_factory(lambda: None, "run")


def test_gate_can_disable_without_importing_sglang(monkeypatch):
    monkeypatch.setenv("HCU_JIT_FP8_FFI", "0")
    assert hcu_env.resolve("HCU_JIT_FP8_FFI") == ("0", "explicit")
    monkeypatch.setattr(bridge.importlib, "import_module",
                        lambda name: pytest.fail("gate imported a kernel module"))
    bridge.install()


def _child(sources, queue):
    try:
        factory, _, calls = _factory(sources)
        alias = factory
        factory(*_key())
        bridge._install_factory(factory, "per_token_group_quant")
        output = torch.empty(3, 4, dtype=torch.float8_e4m3fn)
        alias(*_key()).per_token_group_quant(torch.ones(3, 4), output, torch.ones(3, 1))
        assert calls[-1][1].data_ptr() == output.data_ptr()
        assert torch.equal(output.float(), torch.ones(3, 4))
        queue.put("ok")
    except BaseException as exc:
        queue.put(repr(exc))


def test_spawn_original_cache_protocol(sources):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_child, args=(sources, queue))
    try:
        child.start()
        result = queue.get(timeout=120)
        child.join(timeout=20)
        assert child.exitcode == 0 and result == "ok", result
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        queue.close()
