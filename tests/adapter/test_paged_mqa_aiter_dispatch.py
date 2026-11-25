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

    def __init__(self, contiguous=True):
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return 8448

    def stride(self, axis):
        assert axis == 0
        return 8448 if self._contiguous else 2015232

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

    def run(contiguous=True):
        return fp32_paged_mqa_logits(
            (TensorMetadata(), None), TensorMetadata(contiguous), TensorMetadata(),
            torch.ones((1, 1), dtype=torch.int32), torch.zeros((1, 1), dtype=torch.int32),
            None, 64,
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

