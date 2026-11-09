# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HYV4 CT-int8 scale naming through the original loaders, without source writes.

Contract:
1. CT int8 model (no *_inv params): community load_weights' "_inv" suffix is
   undone before the mixin loader sees the names;
2. fp8 block model (*_inv params present): names pass through unchanged;
3. installation is class-local and idempotent; the mixin, community
   load_weights and cached aliases keep their function objects;
4. drift (own do_load_weights / signature change) is rejected before mutation;
5. gate off -> install returns before importing sglang; install never touches
   a source file.
"""

import __future__
import ast
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from hcu_adapter import hyv4_ct_scale_patch as bridge

# Community HYV4ForCausalLM.load_weights shape (v0522 hunyuan_v4.py:787).
_SOURCE = """def load_weights(self, weights):
    def mapped_weights():
        for name, loaded_weight in weights:
            if name.startswith("model.mtp_layers."):
                continue
            loaded_weight = permute_hyv4_indexer_weight(
                name, loaded_weight, self.config
            )
            if name.endswith((".hc_fn", ".hc_head_fn")):
                name += ".weight"
            if name.endswith(".weight_scale"):
                name += "_inv"
            yield name, loaded_weight

    self.do_load_weights(mapped_weights())
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
        "HEAD:python/sglang/srt/models/hunyuan_v4.py",
    ]).decode()
    cls = next(n for n in ast.parse(text).body
               if isinstance(n, ast.ClassDef) and n.name == "HYV4ForCausalLM")
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "load_weights")
    return inspect.cleandoc("\n" + ast.get_source_segment(text, method, padded=True))


class _Mixin:
    """Stand-in for DeepseekV2WeightLoaderMixin.do_load_weights."""

    def do_load_weights(self, weights, is_nextn=False):
        params = dict(self.named_parameters())
        for name, weight in weights:
            params[name].append(weight)  # KeyError like the real loader
        self.loaded_nextn = is_nextn


def _classes(source):
    ns = {"permute_hyv4_indexer_weight": lambda name, w, config: w}
    # Community annotations (Iterable[Tuple[str, torch.Tensor]]) stay unevaluated.
    flags = __future__.annotations.compiler_flag
    exec(compile(source, "<community-hyv4>", "exec", flags=flags, dont_inherit=True), ns)
    model_cls = type("HYV4ForCausalLM", (_Mixin,), {"load_weights": ns["load_weights"]})
    other_cls = type("DeepseekV2ForCausalLM", (_Mixin,), {})
    return model_cls, other_cls


def _model(cls, params):
    model = cls()
    model.config = None
    store = {p: [] for p in params}
    model.named_parameters = lambda: store.items()
    return model, store


_CT = ["model.layers.0.mlp.shared_experts.gate_up_proj.weight",
       "model.layers.0.mlp.shared_experts.gate_up_proj.weight_scale"]
_FP8 = ["model.layers.0.mlp.shared_experts.gate_up_proj.weight",
        "model.layers.0.mlp.shared_experts.gate_up_proj.weight_scale_inv"]
_CKPT = [("model.layers.0.mlp.shared_experts.gate_up_proj.weight", 1),
         ("model.layers.0.mlp.shared_experts.gate_up_proj.weight_scale", 2),
         ("model.mtp_layers.0.x.weight_scale", 3)]


def test_community_loader_keyerrors_without_bridge(source):
    model_cls, _ = _classes(source)
    model, _ = _model(model_cls, _CT)
    with pytest.raises(KeyError, match="weight_scale_inv"):
        model.load_weights(iter(_CKPT))


def test_ct_int8_keeps_weight_scale(source):
    model_cls, _ = _classes(source)
    bridge._install_class(model_cls)
    model, store = _model(model_cls, _CT)
    model.load_weights(iter(_CKPT))
    assert store[_CT[0]] == [1] and store[_CT[1]] == [2]
    assert model.loaded_nextn is False


def test_fp8_block_still_gets_inv(source):
    model_cls, _ = _classes(source)
    bridge._install_class(model_cls)
    model, store = _model(model_cls, _FP8)
    model.load_weights(iter(_CKPT))
    assert store[_FP8[1]] == [2]


def test_class_local_idempotent_and_aliases_kept(source):
    model_cls, other_cls = _classes(source)
    mixin_fn = _Mixin.do_load_weights
    load_fn = model_cls.load_weights
    alias = model_cls.do_load_weights  # cached before install
    assert bridge._install_class(model_cls) is model_cls
    installed = model_cls.__dict__["do_load_weights"]
    assert bridge._install_class(model_cls) is model_cls
    assert model_cls.__dict__["do_load_weights"] is installed
    assert getattr(installed, bridge._MARKER) is mixin_fn
    assert _Mixin.do_load_weights is mixin_fn and model_cls.load_weights is load_fn
    assert "do_load_weights" not in other_cls.__dict__
    assert inspect.signature(installed) == inspect.signature(mixin_fn)
    assert alias is mixin_fn


def test_nextn_flag_forwarded():
    model_cls, _ = _classes(_SOURCE)
    bridge._install_class(model_cls)
    model, store = _model(model_cls, _CT)
    model.do_load_weights(iter(_CKPT[:2]), is_nextn=True)
    assert model.loaded_nextn is True and store[_CT[1]] == [2]


def test_own_do_load_weights_rejected():
    cls = type("HYV4ForCausalLM", (_Mixin,), {"do_load_weights": lambda s, w, is_nextn=False: None})
    original = cls.__dict__["do_load_weights"]
    with pytest.raises(RuntimeError, match="upstream loader changed"):
        bridge._install_class(cls)
    assert cls.__dict__["do_load_weights"] is original


def test_signature_drift_rejected():
    base = type("M", (), {"do_load_weights": lambda self, weights: None})
    cls = type("HYV4ForCausalLM", (base,), {})
    with pytest.raises(RuntimeError, match="signature changed"):
        bridge._install_class(cls)
    assert "do_load_weights" not in cls.__dict__


def test_public_install_gate_and_no_source_writes(source, monkeypatch, tmp_path):
    from hcu_adapter import hcu_env, registry

    path = tmp_path / "hunyuan_v4.py"
    path.write_text(source, encoding="utf-8")
    path.chmod(0o444)
    before = hashlib.sha256(path.read_bytes()).digest()
    model_cls, _ = _classes(source)
    nextn_cls = type("HYV4ForCausalLMNextN", (_Mixin,), {})
    for name in ("sglang", "sglang.srt", "sglang.srt.models"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for (mod_name, cls_name), cls in zip(bridge._TARGETS, (model_cls, nextn_cls)):
        module = types.ModuleType(mod_name)
        module.__file__ = str(path)
        setattr(module, cls_name, cls)
        monkeypatch.setitem(sys.modules, mod_name, module)
    records = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: records.append(k))
    try:
        monkeypatch.setattr(hcu_env, "enabled", lambda name: False)
        bridge.install()
        assert "do_load_weights" not in model_cls.__dict__ and not records
        monkeypatch.setattr(hcu_env, "enabled", lambda name: True)
        bridge.install()
        bridge.install()
        model, store = _model(model_cls, _CT)
        model.load_weights(iter(_CKPT))
        assert store[_CT[1]] == [2]
        assert "do_load_weights" in nextn_cls.__dict__
        assert len(records) == 4
        assert all(r["mechanism"] == registry.MECH_REBIND for r in records)
        assert hashlib.sha256(path.read_bytes()).digest() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    finally:
        path.chmod(0o644)
