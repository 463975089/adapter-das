# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Speculative metadata allocation must preserve the original scheduling body."""

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

from hcu_adapter import spec_h2d_pin as bridge


_SOURCE = """def eagle_prepare_for_decode(batch):
    batch.maybe_evict_swa()
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.cumulate_penalty_output_tokens()
    page_size = batch.token_to_kv_pool_allocator.page_size
    double_alloc = get_alloc_reserve_per_decode()
    cur, nxt, num_needed_tokens = page_aligned_decode_alloc_lens(
        batch.reqs, reserve=double_alloc, page_size=page_size)
    for r in batch.reqs:
        r.decode_batch_idx += 1
    cur_kv_lens_cpu = torch.tensor(cur, dtype=torch.int32, device="cpu")
    nxt_kv_lens_cpu = torch.tensor(nxt, dtype=torch.int32, device="cpu")
    if page_size > 1 and (get_spec().speculative_eagle_topk or 1) > 1:
        max_alloc_len = int(nxt_kv_lens_cpu.max())
        row_width = batch.req_to_token_pool.req_to_token.shape[1]
        assert max_alloc_len <= row_width, "draft over-allocation"
    cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
    alloc_for_spec_decode(
        batch.tree_cache, batch.req_to_token_pool, reqs=batch.reqs,
        req_pool_indices=batch.req_pool_indices,
        cur_kv_lens=cur_kv_lens_device, nxt_kv_lens=nxt_kv_lens_device,
        cur_kv_lens_cpu=cur_kv_lens_cpu, nxt_kv_lens_cpu=nxt_kv_lens_cpu,
        num_needed_tokens=num_needed_tokens, batch=batch)
"""


def _case(source, filename="<spec-metadata>", unavailable=False):
    allocations, results = [], []

    def tensor(*args, **kwargs):
        allocations.append(dict(kwargs))
        pin = kwargs.pop("pin_memory", False)
        if pin and unavailable:
            raise RuntimeError("no pinned allocator")
        return torch.tensor(*args, **kwargs)

    namespace = {
        "torch": types.SimpleNamespace(tensor=tensor, int32=torch.int32),
        "get_alloc_reserve_per_decode": lambda: 4,
        "get_spec": lambda: types.SimpleNamespace(speculative_eagle_topk=2),
        "page_aligned_decode_alloc_lens": lambda *a, **k: ([3, 7], [8, 12], 10),
        "alloc_for_spec_decode": lambda *a, **k: results.append((a, k)),
    }
    exec(compile("from __future__ import annotations\n" + source, filename, "exec"),
         namespace)
    batch = types.SimpleNamespace(
        maybe_evict_swa=lambda: results.append("evict"),
        cumulate_penalty_output_tokens=lambda: results.append("penalty"),
        sampling_info=types.SimpleNamespace(
            penalizer_orchestrator=types.SimpleNamespace(is_required=True)),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=4),
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=types.SimpleNamespace(shape=(2, 32))),
        reqs=[types.SimpleNamespace(decode_batch_idx=0) for _ in range(2)],
        device="cpu", tree_cache=object(), req_pool_indices=object(),
    )
    return namespace, batch, allocations, results


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
        "HEAD:python/sglang/srt/speculative/eagle_utils.py",
    ]).decode()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "eagle_prepare_for_decode")
    return ast.get_source_segment(text, node)


def test_original_body_aliases_values_and_readonly_source(source, tmp_path):
    path = tmp_path / "community_eagle.py"
    path.write_text(source, encoding="utf-8")
    path.chmod(0o444)
    before = hashlib.sha256(path.read_bytes()).digest()
    try:
        ns, batch, allocations, results = _case(source, str(path))
        alias = ns["eagle_prepare_for_decode"]
        signature, code, original_torch = inspect.signature(alias), alias.__code__, ns["torch"]
        assert bridge._install_function(alias) is alias
        assert bridge._install_function(alias) is alias
        assert inspect.signature(alias) == signature
        assert getattr(alias, bridge._MARKER).__code__ is code
        alias(batch=batch)
        assert ns["torch"] is original_torch
        assert [a["pin_memory"] for a in allocations] == [True, True]
        assert results[:2] == ["evict", "penalty"]
        args, kw = results[2]
        assert args == (batch.tree_cache, batch.req_to_token_pool)
        assert kw["batch"] is batch and kw["reqs"] is batch.reqs
        assert kw["req_pool_indices"] is batch.req_pool_indices
        assert kw["num_needed_tokens"] == 10
        for name, values in (("cur", [3, 7]), ("nxt", [8, 12])):
            assert kw[name + "_kv_lens_cpu"].dtype == torch.int32
            assert kw[name + "_kv_lens_cpu"].tolist() == values
            assert kw[name + "_kv_lens"].tolist() == values
        assert all(r.decode_batch_idx == 1 for r in batch.reqs)
        assert hashlib.sha256(path.read_bytes()).digest() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
        with pytest.raises(TypeError):
            alias(batch, unknown=True)
    finally:
        path.chmod(0o644)


def test_original_allocation_guard_and_penalty_control(source):
    ns, batch, allocations, results = _case(source)
    alias = ns["eagle_prepare_for_decode"]
    bridge._install_function(alias)
    batch.req_to_token_pool.req_to_token.shape = (2, 10)
    batch.sampling_info.penalizer_orchestrator.is_required = False
    with pytest.raises(AssertionError, match="over-allocation"):
        alias(batch)
    assert results == ["evict"] and len(allocations) == 2


def test_unavailable_pinning_preserves_stock_values_and_warns_once(source, caplog):
    ns, batch, allocations, results = _case(source, unavailable=True)
    bridge._install_function(ns["eagle_prepare_for_decode"])
    ns["eagle_prepare_for_decode"](batch)
    ns["eagle_prepare_for_decode"](batch)
    assert len(allocations) == 5  # one failed pin, four stock allocations
    assert sum(a.get("pin_memory", False) for a in allocations) == 1
    assert caplog.text.count("pinned allocation unavailable") == 1
    assert results[-1][1]["nxt_kv_lens"].tolist() == [8, 12]


@pytest.mark.parametrize("change", ["extra-torch", "missing-tensor", "foreign"])
def test_drift_and_foreign_bindings_fail_before_mutation(change):
    source = _SOURCE
    if change == "extra-torch":
        source += "    torch.zeros(1)\n"
    elif change == "missing-tensor":
        source = source.replace("torch.tensor(nxt,", "other(nxt,")
    ns, _, _, _ = _case(source)
    function = ns["eagle_prepare_for_decode"]
    if change == "foreign":
        ns[bridge._DISPATCH] = object()
    before = function.__code__
    with pytest.raises(RuntimeError):
        bridge._install_function(function)
    assert function.__code__ is before


def test_post_install_rebinding_is_rejected():
    ns, _, _, _ = _case(_SOURCE)
    function = ns["eagle_prepare_for_decode"]
    bridge._install_function(function)
    ns[bridge._DISPATCH] = lambda: None
    with pytest.raises(RuntimeError, match="changed after installation"):
        bridge._install_function(function)


@pytest.mark.parametrize("device,dtype", [("cuda", torch.int32), ("cpu", torch.float32)])
def test_private_allocator_rejects_unrelated_tensor_calls(device, dtype):
    facade = bridge._PinnedMetadata(torch)
    with pytest.raises(RuntimeError, match="contract changed"):
        facade.tensor([1], device=device, dtype=dtype)


def test_invalid_stock_allocation_is_not_hidden(caplog):
    def broken(*a, **k):
        raise RuntimeError("invalid metadata")

    facade = bridge._PinnedMetadata(types.SimpleNamespace(tensor=broken, int32=torch.int32))
    with pytest.raises(RuntimeError, match="invalid metadata"):
        facade.tensor([1], dtype=torch.int32, device="cpu")
    assert "pinned allocation unavailable" not in caplog.text
    assert not facade._unavailable


def test_public_install_and_disable(source, monkeypatch, tmp_path):
    from hcu_adapter import hcu_env, registry

    ns, batch, _, results = _case(source)
    module = types.ModuleType("sglang.srt.speculative.eagle_utils")
    module.eagle_prepare_for_decode = ns["eagle_prepare_for_decode"]
    module.__file__ = str(tmp_path / "absent.py")
    for name in ("sglang", "sglang.srt", "sglang.srt.speculative"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(version=types.SimpleNamespace(hip="6.3")))
    records = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: records.append(k))
    monkeypatch.setattr(hcu_env, "enabled", lambda name: False)
    code = module.eagle_prepare_for_decode.__code__
    bridge.install()
    assert module.eagle_prepare_for_decode.__code__ is code
    monkeypatch.setattr(hcu_env, "enabled", lambda name: True)
    bridge.install()
    bridge.install()
    module.eagle_prepare_for_decode(batch)
    assert results[-1][1]["nxt_kv_lens"].tolist() == [8, 12]
    assert len(records) == 2 and all(r["mechanism"] == registry.MECH_REBIND for r in records)
    assert not list(tmp_path.iterdir())


def test_cuda_install_does_not_import_community(monkeypatch):
    from hcu_adapter import hcu_env

    monkeypatch.setattr(hcu_env, "enabled", lambda name: True)
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(version=types.SimpleNamespace(hip=None)))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.eagle_utils", None)
    bridge.install()


def _spawn_worker(source, queue):
    ns, batch, allocations, results = _case(source)
    alias = ns["eagle_prepare_for_decode"]
    bridge._install_function(alias)
    alias(batch)
    queue.put((results[-1][1]["cur_kv_lens"].tolist(),
               all(a.get("pin_memory") for a in allocations)))


def test_spawned_interpreter_installs_its_own_binding(source):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_worker, args=(source, queue))
    process.start()
    try:
        assert queue.get(timeout=40) == ([3, 7], True)
        process.join(timeout=20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        queue.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires explicit remote GPU")
def test_original_prepare_with_real_h2d(source):
    ns, batch, _, results = _case(source)
    ns["torch"] = torch
    batch.device = "cuda"
    alias = ns["eagle_prepare_for_decode"]
    bridge._install_function(alias)
    alias(batch)
    kw = results[-1][1]
    for name, expected in (("cur", [3, 7]), ("nxt", [8, 12])):
        cpu, gpu = kw[name + "_kv_lens_cpu"], kw[name + "_kv_lens"]
        assert cpu.is_pinned() and cpu.dtype == torch.int32
        assert gpu.is_cuda and gpu.dtype == torch.int32
        assert gpu.cpu().tolist() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires explicit remote GPU")
@pytest.mark.parametrize("path", ["allocator", "compat-helper"])
def test_async_pinned_sources_are_safe_beyond_old_ring_depth(path):
    # Retain only device results. Many H2D copies on independent streams must
    # not overwrite pinned source storage before earlier DMA has consumed it.
    facade = bridge._PinnedMetadata(torch)
    streams = [torch.cuda.Stream() for _ in range(3)]
    outputs = []
    for wave in range(48):
        with torch.cuda.stream(streams[wave % len(streams)]):
            values = [wave * 100 + (i % 97) for i in range(8192 + wave * 17)]
            if path == "allocator":
                cpu = facade.tensor(values, dtype=torch.int32, device="cpu")
                assert cpu.is_pinned()
                gpu = cpu.to("cuda", non_blocking=True)
            else:
                cpu = torch.tensor(values, dtype=torch.int32)
                gpu = bridge.pinned_h2d(cpu, "cuda", "shared-key")
            outputs.append((gpu, values))
            del cpu
    torch.cuda.synchronize()
    for device, expected in outputs:
        assert device.cpu().tolist() == expected
