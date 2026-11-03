"""Compiler-input provenance, alias retention and original community loader seams."""

import ast
import hashlib
import io
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import types

import pytest

from hcu_adapter import jit_sources as bridge


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()
    }


def _synthetic(root):
    from hcu_adapter import (
        jit_rope, jit_store_cache, jit_ptgq, jit_dpskmeta, jit_quant_fp8,
        jit_warp, jit_concat_mla, jit_dot_product,
    )
    files = {}

    def add(name, text):
        files.setdefault(name, []).append(text)

    for name, needle, _ in jit_rope._PATCHES:
        add("csrc/" + name, needle)
    for module in (jit_store_cache, jit_concat_mla):
        for needle, _ in module._PATCHES:
            add("csrc/" + module._RELPATH, needle)
    add("csrc/" + jit_concat_mla._RELPATH, jit_concat_mla._DLDEVICE_NEEDLE)
    add("csrc/" + jit_ptgq._RELPATH, jit_ptgq._NEEDLE)
    for name, pairs in jit_dpskmeta._PATCHES.items():
        for needle, _ in pairs:
            add("csrc/" + name, needle)
    for name, needle, _ in jit_quant_fp8._PATCHES + jit_quant_fp8._PATCHES_V2:
        add("csrc/" + name, needle)
    for _, needle, _ in jit_quant_fp8._PATCHES_WARP:
        add(jit_warp._RELPATH, needle)
    for name, needle, _ in jit_quant_fp8._PATCHES_INCLUDE:
        add("include/sgl_kernel/" + name, needle)
    for needle, _ in jit_warp._SHFL_MAP:
        add(jit_warp._RELPATH, needle)
    for needle, _ in jit_dot_product._PATCHES:
        add(jit_dot_product._RELPATH, needle)
    for name, chunks in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(chunks))


def _git(tree, *args):
    return subprocess.check_output([
        "git", "-c", f"safe.directory={tree}", "-C", str(tree), *args,
    ])


@pytest.fixture(params=["offline", "community-git"])
def source(tmp_path, request):
    root = tmp_path / "community" / "jit"
    root.mkdir(parents=True)
    if request.param == "offline":
        _synthetic(root)
        return root, None
    tree = os.environ.get("HCU_TEST_SGLANG_TREE")
    if not tree:
        pytest.skip("requires explicitly selected community Git tree")
    prefix = "python/sglang/kernels/jit/"
    archive = _git(tree, "archive", "HEAD", prefix + "csrc", prefix + "include")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = Path(member.name[len(prefix):])
            assert not rel.is_absolute() and ".." not in rel.parts
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    return root, tree


def test_complete_bundle_preserves_sources_and_all_transforms(source, tmp_path):
    root, _ = source
    before = _snapshot(root)
    bundle, manifest = bridge.prepare(root, tmp_path / "cache")
    assert _snapshot(root) == before
    assert bundle.parent == tmp_path / "cache"
    inputs = {k: v for k, v in before.items() if Path(k).suffix not in (".py", ".pyc", ".pyo")}
    assert manifest["recipe"]["inputs"] == inputs
    assert not list(bundle.rglob("*.py"))
    assert any(inputs[k] != manifest["outputs"][k] for k in inputs)
    for _, record in manifest["records"]:
        rel = record["target"].removeprefix("jit-input:sglang/kernels/jit/")
        assert (bundle / rel).is_file(), record["target"]
    warp = (bundle / "include/sgl_kernel/warp.cuh").read_text()
    assert "T _v;" in warp and "__hip_bfloat162 _v;" not in warp
    assert "__shfl_up_sync(active_mask" not in warp
    assert (bundle / "cuda-headers/cuda_fp8.h").is_file()
    for primitive in ("load", "store", "scan"):
        body = (bundle / f"cuda-headers/cub/block/block_{primitive}.cuh").read_text()
        assert f"#include <hipcub/block/block_{primitive}.hpp>" in body
        assert "namespace cub = hipcub;" in body
    again, second = bridge.prepare(root, tmp_path / "cache")
    assert again == bundle and second == manifest
    assert not list((tmp_path / "cache").glob(".staging-*"))


def test_identity_changes_for_input_and_transformation(source, tmp_path, monkeypatch):
    root, _ = source
    first, _ = bridge.prepare(root, tmp_path / "cache")
    (root / "include/new-transitive.cuh").write_text("#define TEST_VALUE 1\n")
    second, _ = bridge.prepare(root, tmp_path / "cache")
    assert first != second
    old_digest = bridge._digest
    monkeypatch.setattr(bridge, "_digest", lambda data: old_digest(b"new-recipe" + data))
    third, _ = bridge.prepare(root, tmp_path / "cache")
    assert second != third


def test_corrupt_bundle_refused_without_repairing_sources(source, tmp_path):
    root, _ = source
    bundle, _ = bridge.prepare(root, tmp_path / "cache")
    target = bundle / "include/sgl_kernel/warp.cuh"
    target.write_text("corrupt")
    before = _snapshot(root)
    with pytest.raises(RuntimeError, match="invalid compiler bundle"):
        bridge.prepare(root, tmp_path / "cache")
    assert target.read_text() == "corrupt" and _snapshot(root) == before


def test_failed_transform_publishes_nothing(source, tmp_path, monkeypatch):
    root, _ = source
    before = _snapshot(root)
    from hcu_adapter import jit_dot_product

    def fail(stage, register):
        assert stage != root
        (stage / "include/partial.cuh").write_text("partial")
        raise RuntimeError("transform failed")

    monkeypatch.setattr(jit_dot_product, "_apply", fail)
    with pytest.raises(RuntimeError, match="transform failed"):
        bridge.prepare(root, tmp_path / "cache")
    assert not list((tmp_path / "cache").iterdir())
    assert _snapshot(root) == before


def test_cache_cannot_be_in_community_repository(source, tmp_path):
    root, _ = source
    (root.parent / ".git").write_text("gitdir: somewhere")
    with pytest.raises(RuntimeError, match="outside"):
        bridge.prepare(root, root.parent / "cache")


def _publish_child(root, cache, queue):
    try:
        root = Path(root).resolve()

        def reject_source_write(event, args):
            paths = ()
            if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
                if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                    paths = (args[0],)
            elif event in ("os.rename", "os.remove", "os.rmdir", "os.mkdir"):
                paths = args[:2] if event == "os.rename" else args[:1]
            for value in paths:
                if isinstance(value, (str, os.PathLike)):
                    path = Path(value).resolve()
                    if path == root or root in path.parents:
                        raise RuntimeError("community source write forbidden by test audit")

        sys.addaudithook(reject_source_write)
        try:
            (root / "forbidden-test.cuh").write_text("probe")
        except RuntimeError as exc:
            assert "forbidden by test audit" in str(exc)
        else:
            raise AssertionError("source write audit not armed")
        bundle, _ = bridge.prepare(root, cache)
        queue.put(("ok", str(bundle)))
    except BaseException as exc:
        queue.put(("error", repr(exc)))


def test_spawn_concurrent_publish(source, tmp_path):
    root, _ = source
    before = _snapshot(root)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    children = [context.Process(
        target=_publish_child, args=(root, tmp_path / "cache", queue),
    ) for _ in range(2)]
    try:
        for child in children:
            child.start()
        results = [queue.get(timeout=120) for _ in children]
        for child in children:
            child.join(timeout=30)
            assert child.exitcode == 0
        assert all(status == "ok" for status, _ in results), results
        assert results[0] == results[1]
        assert len(list((tmp_path / "cache").iterdir())) == 1
        assert _snapshot(root) == before
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()
        queue.close()


def test_symbolic_source_assets_are_refused(source, tmp_path):
    root, _ = source
    try:
        (root / "include/alias.cuh").symlink_to(root / "include/sgl_kernel/warp.cuh")
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(RuntimeError, match="symbolic source asset"):
        bridge.prepare(root, tmp_path / "cache")


class _SpecCaptured(Exception):
    pass


def _functions(root, tree):
    import pathlib
    resolver_text = """def resolve_sources(files):
    return tuple(str(p.resolve() if p.is_absolute() else
                     (KERNEL_PATH / "csrc" / p).resolve())
                 for p in map(pathlib.Path, files or []))
"""
    loader_text = """def load_jit(*args, cpp_files=None, cuda_files=None, extra_include_paths=None,
                  extra_dependencies=None):
    includes = list(DEFAULT_INCLUDE) + (extra_include_paths or [])
    for dep in extra_dependencies or []:
        includes += REGISTERED_DEPENDENCIES[dep]()
    spec = BuildSpec(cpp_files=resolve_sources(cpp_files),
                     cuda_files=resolve_sources(cuda_files),
                     include_paths=tuple(includes))
    return ninja.generate(spec)
"""
    if tree:
        prefix = "HEAD:python/sglang/kernels/jit/utils/compile/"

        def extract(file, name):
            text = _git(tree, "show", prefix + file).decode()
            node = next(n for n in ast.parse(text).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            return ast.get_source_segment(text, node)

        resolver_text = extract("spec.py", "resolve_sources")
        loader_text = extract("loader.py", "load_jit")
    ns = {"pathlib": pathlib, "KERNEL_PATH": root}
    exec("from __future__ import annotations\n" + resolver_text, ns)
    resolver = ns["resolve_sources"]
    captured = []

    def generate(spec):
        captured.append(spec)
        raise _SpecCaptured

    false_env = types.SimpleNamespace(get=lambda: False)
    ls = {
        "resolve_sources": resolver, "DEFAULT_INCLUDE": [str(root / "include")],
        "BuildSpec": lambda **kw: types.SimpleNamespace(**kw),
        "is_hip_runtime": lambda: True,
        "envs": types.SimpleNamespace(SGLANG_JIT_LOG_RESOURCE_USAGE=false_env),
        "REGISTERED_DEPENDENCIES": {}, "get_default_target_flags": lambda: [],
        "DEFAULT_CFLAGS": [], "DEFAULT_LDFLAGS": [],
        "ninja": types.SimpleNamespace(generate=generate),
    }
    exec("from __future__ import annotations\n" + loader_text, ls)
    return resolver, ls["load_jit"], captured


def test_original_loader_and_cached_aliases_route_all_compiler_inputs(source, tmp_path):
    root, tree = source
    bundle, _ = bridge.prepare(root, tmp_path / "cache")
    resolver, loader, captured = _functions(root, tree)
    cached_resolver, cached_loader = resolver, loader
    signatures = (inspect.signature(resolver), inspect.signature(loader))
    source_file = root / "csrc/elementwise/rope.cuh"
    outside = tmp_path / "external.cu"
    outside.write_text("// external source")
    bridge.bind(resolver, loader, root, bundle)
    assert resolver is cached_resolver and loader is cached_loader
    assert (inspect.signature(resolver), inspect.signature(loader)) == signatures
    assert resolver(None) == ()
    assert resolver(["elementwise/rope.cuh", source_file, outside]) == (
        str(bundle / "csrc/elementwise/rope.cuh"),
        str(bundle / "csrc/elementwise/rope.cuh"), str(outside),
    )
    assert resolver(["../include/sgl_kernel/warp.cuh"]) == (
        str(bundle / "include/sgl_kernel/warp.cuh"),
    )
    with pytest.raises(_SpecCaptured):
        cached_loader("test", cuda_files=[str(source_file)],
                      cpp_files=[str(outside)],
                      extra_include_paths=[str(root / "include/sgl_kernel"), str(tmp_path)])
    spec = captured[-1]
    assert spec.cuda_files == (str(bundle / "csrc/elementwise/rope.cuh"),)
    assert spec.cpp_files == (str(outside),)
    assert spec.include_paths == (
        str(bundle / "include"), str(bundle / "cuda-headers"),
        str(bundle / "include/sgl_kernel"), str(tmp_path),
    )
    loader.__globals__["REGISTERED_DEPENDENCIES"]["late-provider"] = lambda: [
        str(root / "include/sgl_kernel"), str(tmp_path),
    ]
    with pytest.raises(_SpecCaptured):
        cached_loader("provider", extra_dependencies=["late-provider"])
    assert captured[-1].include_paths == (
        str(bundle / "include"), str(bundle / "cuda-headers"),
        str(bundle / "include/sgl_kernel"), str(tmp_path),
    )
    bridge.bind(resolver, loader, root, bundle)
    assert loader.__globals__["DEFAULT_INCLUDE"] == [str(root / "include")]
    assert resolver.__globals__["KERNEL_PATH"] == root
    # A foreign replacement must be reported rather than silently ignored.
    loader.__code__ = (lambda *a, **k: None).__code__
    with pytest.raises(RuntimeError, match="changed after"):
        bridge.bind(resolver, loader, root, bundle)


def test_preflight_failure_leaves_both_entries_unmodified(source, tmp_path):
    root, tree = source
    resolver, loader, _ = _functions(root, tree)
    codes = (resolver.__code__, loader.__code__)
    loader.__globals__["DEFAULT_INCLUDE"] = []
    with pytest.raises(RuntimeError, match="include root changed"):
        bridge.bind(resolver, loader, root, tmp_path / "bundle")
    assert (resolver.__code__, loader.__code__) == codes
    assert not hasattr(resolver, bridge._MARKER)


def test_actual_host_compiler_and_transitive_cache_invalidation(source, tmp_path):
    """Real official BuildSpec/ninja/cache/tvm-ffi, without exposing a GPU."""
    if os.environ.get("HCU_TEST_JIT_COMPILER") != "1":
        pytest.skip("requires explicitly selected remote community compiler")
    from sglang.kernels.jit.utils.compile import loader as official, spec as spec_module

    root, _ = source
    (root / "csrc/adapter_compile_probe.cuh").write_text(
        "#include <adapter_transitive_probe.h>\n"
        "namespace sglang { int adapter_compile_probe(int x) { return x + ADAPTER_PROBE; } }\n"
    )
    header = root / "include/adapter_transitive_probe.h"
    bundles = []
    for delta in (7, 19):
        header.write_text(f"#define ADAPTER_PROBE {delta}\n")
        before = _snapshot(root)
        # Fresh original function objects for each fixture root. Actual
        # community modules, origin and Python files are never redirected.
        original_resolve = spec_module.resolve_sources
        resolver = types.FunctionType(
            original_resolve.__code__, dict(original_resolve.__globals__, KERNEL_PATH=root),
            original_resolve.__name__, original_resolve.__defaults__,
        )
        original_load = inspect.unwrap(official.load_jit)
        loader = types.FunctionType(
            original_load.__code__, dict(
                original_load.__globals__, resolve_sources=resolver,
                DEFAULT_INCLUDE=[str(root / "include")],
            ), original_load.__name__, original_load.__defaults__,
        )
        loader.__kwdefaults__ = original_load.__kwdefaults__
        bundle, _ = bridge.prepare(root, tmp_path / "cache")
        bundles.append(bundle)
        bridge.bind(resolver, loader, root, bundle)
        module = loader(
            "adapter_host_compile_probe",
            cpp_files=["adapter_compile_probe.cuh"],
            cpp_wrappers=[("probe", "adapter_compile_probe")],
        )
        assert module.probe(5) == 5 + delta
        assert _snapshot(root) == before
    assert bundles[0] != bundles[1]


@pytest.mark.parametrize("kernel", ["rope-neox", "rope-interleaved", "quant32", "quant128"])
def test_offline_hip_template_compilation(source, tmp_path, kernel):
    """Instantiate actual HIP wrappers with an explicit target, without a GPU."""
    root, tree = source
    if not tree or os.environ.get("HCU_TEST_JIT_HIP_SYNTAX") != "1":
        pytest.skip("requires community assets and explicit remote HIP compile target")
    import torch
    from sglang.kernels.jit.utils import make_cpp_args
    from sglang.kernels.jit.utils.compile import spec, toolchain

    bundle, _ = bridge.prepare(root, tmp_path / "cache")
    if kernel.startswith("rope"):
        args = make_cpp_args(kernel == "rope-neox", 64, False,
                             torch.bfloat16, torch.bfloat16)
        file = "elementwise/rope.cuh"
        wrappers = (("run", f"FusedRopeKernel<{args}>::run"),)
    else:
        group = int(kernel.removeprefix("quant"))
        args = make_cpp_args(torch.bfloat16, torch.float8_e4m3fn, group,
                             False, True, True, False, False)
        file = "gemm/per_token_group_quant.cuh"
        wrappers = (("run", f"PerTokenGroupQuantFlatKernel<{args}>::run"),)
    unit = tmp_path / "hip_template_probe.cu"
    unit.write_text(spec._wrapper_source((str(bundle / "csrc" / file),), wrappers))
    includes = toolchain.base_include_paths() + [
        str(bundle / "include"), str(bundle / "cuda-headers"),
    ]
    before = _snapshot(root)
    result = subprocess.run([
        toolchain.device_compiler_path(), "--offload-arch=gfx938", "-fsyntax-only",
        "-std=c++20", "-O3", "-DUSE_ROCM", "-DHIP_FP8_TYPE_E4M3=1",
        *toolchain.base_cuda_flags(), *["-I" + p for p in includes], str(unit),
    ], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _snapshot(root) == before


def _gpu():
    if os.environ.get("HCU_TEST_JIT_GPU") != "1":
        pytest.skip("requires explicit remote HIP compiler validation")
    import torch
    assert torch.version.hip and torch.cuda.is_available()
    return torch


@pytest.mark.parametrize("neox", [True, False])
def test_gpu_original_rope_compilation_and_variable_graph_replay(neox):
    torch = _gpu()
    bundle = bridge.install()
    from sglang.kernels.ops.attention.rope import _jit_fused_rope_module
    module = _jit_fused_rope_module(neox, 64, torch.bfloat16, torch.bfloat16)
    assert bundle.is_dir()
    generator = torch.Generator(device="cuda").manual_seed(41)
    q = torch.empty((7, 4, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((7, 2, 64), device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((128, 32), generator=generator, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    pos = torch.arange(7, device="cuda", dtype=torch.int64)

    def run():
        module.run_rope(q, k, cache, pos)

    q.zero_()
    k.zero_()
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    def reference(x):
        cos, sin = cache[pos, :32, None], cache[pos, 32:, None]
        cos, sin = cos.transpose(1, 2), sin.transpose(1, 2)
        a, b = x.float().chunk(2, -1) if neox else (x.float()[..., ::2], x.float()[..., 1::2])
        left, right = a * cos - b * sin, b * cos + a * sin
        return (torch.cat((left, right), -1) if neox
                else torch.stack((left, right), -1).flatten(-2))

    for wave in range(3):
        qin = torch.randn(q.shape, generator=generator, device="cuda", dtype=q.dtype)
        kin = torch.randn(k.shape, generator=generator, device="cuda", dtype=k.dtype)
        q.copy_(qin)
        k.copy_(kin)
        pos.copy_(torch.arange(7, device="cuda") + wave * 11)
        graph.replay()
        for actual, original in ((q, qin), (k, kin)):
            want = reference(original)
            torch.testing.assert_close(actual.float(), want, atol=.032, rtol=.008)
            assert torch.nn.functional.cosine_similarity(
                actual.float().flatten(), want.flatten(), dim=0).item() >= .9999


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("backend", ["generic", "v2"])
def test_gpu_original_quant_compilation_and_variable_graph_replay(group_size, backend):
    torch = _gpu()
    bridge.install()
    from importlib import import_module
    from hcu_adapter.jit_fp8_ffi import _OutputStorage
    if backend == "generic":
        api = import_module("sglang.kernels.ops.quantization.per_token_group_quant")
        module = api._jit_module(torch.bfloat16, torch.float8_e4m3fn, group_size,
                                 False, True, True, False, False, False)
    else:
        api = import_module("sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2")
        module = api._jit_module(torch.bfloat16, torch.float8_e4m3fn, False)
    assert isinstance(module, _OutputStorage)
    x = torch.ones((9, 256), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((9, 256 // group_size), device="cuda", dtype=torch.float32)

    def run():
        if backend == "generic":
            api.per_token_group_quant(x, output_q=q, output_s=s, group_size=group_size)
        else:
            api.per_token_group_quant_8bit_v2(x, q, s, group_size, 1e-10, -448., 448.)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    generator = torch.Generator(device="cuda").manual_seed(83)
    for magnitude in (.25, 2., 17.):
        x.copy_(torch.randn(x.shape, generator=generator, device="cuda") * magnitude)
        graph.replay()
        groups = x.float().reshape(9, -1, group_size)
        ref_s = groups.abs().amax(-1).clamp_min(1e-10) / 448.
        ref_q = (groups / ref_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(s, ref_s, atol=1e-7, rtol=2e-5)
        # Near rounding boundaries, compare dequantized values to the original
        # FP32 input with one FP8 bin allowance, plus a tight cosine condition.
        dequant = (q.float().reshape_as(groups) * s[..., None]).flatten()
        want = (ref_q.float() * ref_s[..., None]).flatten()
        torch.testing.assert_close(dequant, want, atol=magnitude * .02, rtol=.13)
        assert torch.nn.functional.cosine_similarity(dequant, want, dim=0) >= .9999
        assert torch.nn.functional.cosine_similarity(dequant, groups.flatten(), dim=0) >= .999
