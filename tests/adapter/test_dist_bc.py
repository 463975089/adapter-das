"""Broadcast compatibility must preserve aliases, protocol and source bytes."""

import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from hcu_adapter import dist_bc


def function(namespace=None):
    scope = {"np": np, **(namespace or {})}
    exec(
        "def broadcast_pyobj(data, rank=0, *, force_cpu_device=True):\n"
        "    array = np.frombuffer(data, dtype=np.uint8)\n"
        "    if not array.flags.writeable:\n"
        "        raise BufferError('readonly DLPack input')\n"
        "    return array, rank, force_cpu_device\n",
        scope,
    )
    return scope["broadcast_pyobj"]


def test_cached_alias_and_default_keyword_contract_survive():
    original = function()
    alias = original
    signature = inspect.signature(original)
    with pytest.raises(BufferError):
        alias(b"abc")
    assert dist_bc._install_function(original) is alias
    assert inspect.signature(alias) == signature
    array, rank, force_cpu = alias(b"abc", 3, force_cpu_device=False)
    assert array.tobytes() == b"abc" and array.flags.writeable
    assert rank == 3 and force_cpu is False
    assert not np.frombuffer(b"abc", dtype=np.uint8).flags.writeable
    assert original.__globals__["np"] is np  # other common functions unchanged
    assert dist_bc._install_function(original) is alias
    with pytest.raises(TypeError):
        alias(b"abc", unexpected=True)


def test_writable_buffers_keep_their_storage():
    data = bytearray(b"abc")
    array = dist_bc._WritableBuffers(np).frombuffer(data, dtype=np.uint8)
    array[0] = 100
    assert data == b"dbc"


def test_unsupported_or_foreign_bindings_fail_without_mutation():
    bad = lambda data: data
    code = bad.__code__
    with pytest.raises(RuntimeError, match="contract changed"):
        dist_bc._install_function(bad)
    assert bad.__code__ is code
    installed = function()
    dist_bc._install_function(installed)
    installed.__globals__[dist_bc._DISPATCH] = lambda *a, **k: None
    with pytest.raises(RuntimeError, match="changed after"):
        dist_bc._install_function(installed)
    occupied = function({dist_bc._DISPATCH: object()})
    code = occupied.__code__
    with pytest.raises(RuntimeError, match="already exists"):
        dist_bc._install_function(occupied)
    assert occupied.__code__ is code


def test_public_install_needs_no_writable_source(monkeypatch, tmp_path):
    from hcu_adapter import registry

    common = types.ModuleType("sglang.srt.utils.common")
    common.__file__ = str(tmp_path / "nonexistent-readonly-common.py")
    common.broadcast_pyobj = function()
    alias = common.broadcast_pyobj
    modules = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules[common.__name__] = common
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    records = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: records.append((a, k)))
    dist_bc.install()
    dist_bc.install()
    assert alias(b"payload")[0].tobytes() == b"payload"
    assert not list(tmp_path.iterdir())
    assert all(k["mechanism"] == registry.MECH_REBIND for _, k in records)
    assert not any(op == "dist_broadcast" for op, _ in registry.TEXT_PATCH_ALLOWLIST)


def _rank_worker(rank, source, rendezvous):
    """Spawned rank: execute the exact community function, not a protocol copy."""
    import datetime
    import torch.distributed as distributed
    from hcu_adapter.dist_bc import _install_function

    spec = importlib.util.spec_from_file_location("_upstream_broadcast_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    alias = module.broadcast_pyobj
    signature = inspect.signature(alias)
    _install_function(alias)
    assert signature == inspect.signature(alias)
    distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        for src in (0, 1):
            for payload in ([], [{"text": "海光", "bytes": b"\x00\xff", "wave": src}],
                            [list(range(4097)), None, True]):
                result = alias(payload if rank == src else None, rank, src=src)
                assert result == payload
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not os.environ.get("HCU_TEST_SGLANG_TREE"),
                    reason="requires explicit community SGLang tree and CPU Gloo")
def test_pristine_upstream_protocol_with_spawned_gloo_ranks(tmp_path):
    import ast
    import hashlib
    import torch.multiprocessing as multiprocessing

    tree = Path(os.environ["HCU_TEST_SGLANG_TREE"]).resolve()
    raw = subprocess.check_output([
        "git", "-c", f"safe.directory={tree}", "-C", str(tree), "show",
        "HEAD:python/sglang/srt/utils/common.py",
    ]).decode()
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == "broadcast_pyobj")
    body = ast.get_source_segment(raw, node)
    assert "np.frombuffer(serialized_data, dtype=np.uint8)" in body
    assert ".copy()" not in body
    source = tmp_path / "community_broadcast.py"
    source.write_text(
        "from typing import Any, List, Optional\n"
        "import pickle\nimport numpy as np\nimport torch\n"
        "import torch.distributed as dist\n"
        "def is_musa(): return False\n" + body + "\n",
        encoding="utf-8",
    )
    source.chmod(0o444)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        multiprocessing.spawn(
            _rank_worker, args=(str(source), (tmp_path / "rendezvous").as_uri()),
            nprocs=2, join=True,
        )
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    finally:
        source.chmod(0o644)
