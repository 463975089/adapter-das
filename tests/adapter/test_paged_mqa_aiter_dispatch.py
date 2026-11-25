# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch and actual configuration-UI round trips for paged MQA."""

from contextlib import nullcontext
import importlib
import importlib.machinery
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter._base import LibStubFinder, Missing
from hcu_adapter.infra import hcu_config, registry
from hcu_adapter.kernels import mla_attention_emu, vllm_paged_mqa_triton
from hcu_adapter.kernels.vllm_paged_mqa_aiter import fp32_paged_mqa_logits


class TensorMetadata:
    device = SimpleNamespace(type="cuda")
    shape = (1, 1, 32, 128)
    requires_grad = False
    dtype = torch.float8_e4m3fn

    def __init__(self, contiguous=True):
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return 8448

    def stride(self, axis):
        if axis == -1:
            return 1
        assert axis == 0
        return 8448 if self._contiguous else 2015232

    def storage_offset(self):
        return 0

    def data_ptr(self):
        return 128

    def contiguous(self):
        return self


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.setattr(mla_attention_emu, "_validate_fp32_paged_mqa", lambda *args: None)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(gcnArchName="gfx938"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    fallback_calls = []
    result = object()

    def fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(vllm_paged_mqa_triton, "fp32_paged_mqa_logits", fallback)
    monkeypatch.setitem(sys.modules, "hcu_adapter.kernels._paged_mqa_mask_triton", SimpleNamespace(owned_logits_mask=lambda output, lengths: output))

    def run(contiguous=True, rows=1, width=64):
        query = TensorMetadata()
        query.shape = (rows, 1, 32, 128)
        return fp32_paged_mqa_logits(
            (query, None), TensorMetadata(contiguous), TensorMetadata(),
            torch.ones((rows, 1), dtype=torch.int32), torch.zeros((rows, 1), dtype=torch.int32),
            None, width,
        )

    return run, fallback_calls, result


def test_real_missing_finder_uses_fallback(monkeypatch, dispatch):
    for name in list(sys.modules):
        if name == "aiter" or name.startswith("aiter."):
            monkeypatch.delitem(sys.modules, name)
    # Track the initially absent entry so the import-created stub is removed.
    monkeypatch.setitem(sys.modules, "aiter", None)
    del sys.modules["aiter"]
    find_spec = importlib.machinery.PathFinder.find_spec

    def library_missing(name, path=None, target=None):
        if name == "aiter":
            return None
        return find_spec(name, path, target)

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", library_missing)
    monkeypatch.setattr(sys, "meta_path", [LibStubFinder("aiter"), *sys.meta_path])
    api = importlib.import_module("aiter").paged_mqa_logits
    assert isinstance(api, Missing) and callable(api)
    run, calls, expected = dispatch
    assert run() is expected
    assert len(calls) == 1


@pytest.mark.parametrize("kind,contiguous", [("no_export", True), ("old_stride", False), ("new_stride", False), ("old_contiguous", True), ("kernel_error", True)])
def test_native_api_capability_and_error_policy(monkeypatch, dispatch, kind, contiguous):
    native_calls = []
    native_result = object()

    def native(*args, **kwargs):
        native_calls.append((args, kwargs))
        if kind == "kernel_error":
            raise NotImplementedError("native kernel failure")
        return native_result

    native.supports_page_stride = kind == "new_stride"
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace(**({} if kind == "no_export" else {"paged_mqa_logits": native})))
    run, fallback_calls, fallback_result = dispatch
    if kind == "kernel_error":
        with pytest.raises(NotImplementedError, match="native kernel failure"):
            run(contiguous)
        assert len(native_calls) == 1 and not fallback_calls
    elif kind in ("new_stride", "old_contiguous"):
        assert run(contiguous) is native_result
        assert len(native_calls) == 1 and not fallback_calls
    else:
        assert run(contiguous) is fallback_result
        assert len(fallback_calls) == 1 and not native_calls


@pytest.mark.parametrize("backend", ["emu", "aiter", "triton"])
def test_ui_export_load_provider_round_trip(monkeypatch, tmp_path, backend):
    from hcu_adapter.bridges import vllm_paged_mqa as bridge

    root = Path(__file__).resolve().parents[2]
    html = (root / "tools/hcu_config_ui.html").read_text()
    # Execute the actual catalog, defaults, and export function without DOM setup.
    catalog = re.search(r"const CATALOG = (\{.*?\});", html, re.S).group(0)
    defaults = re.search(r"const OP_DEFAULT = (\{.*?\});", html, re.S).group(0)
    exporter = html[html.index("function buildYaml("):html.index("// ---- launch command builder")]
    script = catalog + defaults + exporter + "\nconsole.log(buildYaml({vllm_paged_mqa_logits: process.argv[1]}));"
    result = subprocess.run(["node", "-e", script, backend], capture_output=True, text=True, check=True)
    yaml = result.stdout
    assert ("vllm_paged_mqa_logits:" in yaml) == (backend != "aiter")
    config = tmp_path / "provider.yaml"
    config.write_text(yaml)
    monkeypatch.setenv("HCU_CONFIG", str(config))
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setattr(hcu_config, "_STATE", dict(hcu_config._STATE))
    monkeypatch.setattr(registry, "_RECORDS", dict(registry._RECORDS))
    monkeypatch.setattr(registry, "_BOUND", dict(registry._BOUND))
    hcu_config.load(force=True)
    _, scorer = bridge.get_paged_mqa_provider()
    expected = {
        "emu": bridge.fp8_fp4_paged_mqa_logits,
        "aiter": bridge.aiter_paged_mqa_logits,
        "triton": bridge.triton_paged_mqa_logits,
    }
    assert scorer is expected[backend]



@pytest.mark.parametrize("rows,width,native_allowed", [
    (4, (1 << 30) - 1, True),
    (4, 1 << 30, True),
    (4, (1 << 30) + 1, False),
])
def test_native_output_index_boundary_without_allocation(monkeypatch, dispatch, rows, width, native_allowed):
    native_calls = []
    reference_calls = []
    native_result = object()
    reference_result = object()

    def native(*args, **kwargs):
        native_calls.append((args, kwargs))
        return native_result

    def reference(*args, **kwargs):
        reference_calls.append((args, kwargs))
        return reference_result

    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace(paged_mqa_logits=native))
    monkeypatch.setattr(mla_attention_emu, "fp8_fp4_paged_mqa_logits", reference)
    run, triton_calls, _ = dispatch
    assert run(rows=rows, width=width) is (native_result if native_allowed else reference_result)
    assert len(native_calls) == int(native_allowed)
    assert len(reference_calls) == int(not native_allowed)
    assert not triton_calls


@pytest.mark.parametrize("rows,width", [(2, (1 << 30) + 1), (4, 1 << 30)])
def test_explicit_triton_large_output_uses_reference(monkeypatch, rows, width):
    from hcu_adapter.kernels.vllm_paged_mqa_triton import fp32_paged_mqa_logits as scorer

    query = TensorMetadata()
    query.shape = (rows, 1, 32, 128)
    monkeypatch.setattr(mla_attention_emu, "_validate_fp32_paged_mqa", lambda *args: None)
    calls = []
    expected = object()

    def reference(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(mla_attention_emu, "fp8_fp4_paged_mqa_logits", reference)
    assert scorer((query, None), TensorMetadata(), TensorMetadata(), None, None, None, width) is expected
    assert len(calls) == 1


@pytest.mark.parametrize("width", [(1 << 30) - 1, 1 << 30])
def test_triton_output_index_boundary_without_allocation(monkeypatch, width):
    from hcu_adapter.kernels.vllm_paged_mqa_triton import fp32_paged_mqa_logits as scorer

    query = TensorMetadata()
    query.shape = (2, 1, 32, 128)
    calls = []
    expected = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(mla_attention_emu, "_validate_fp32_paged_mqa", lambda *args: None)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(
        current_platform=SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn)))
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.ops.rocm_aiter_mla_sparse", SimpleNamespace(
        rocm_fp8_paged_mqa_logits_triton=native))
    monkeypatch.setitem(sys.modules, "hcu_adapter.kernels._paged_mqa_mask_triton", SimpleNamespace(
        owned_logits_mask=lambda output, lengths: output))
    tensor = TensorMetadata()
    assert scorer((query, None), tensor, tensor, tensor, tensor, None, width) is expected
    assert len(calls) == 1
