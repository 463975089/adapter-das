"""The scoped capture gate must preserve community forward dispatch semantics."""

import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from hcu_adapter import moe_dualstream_gate as gate


# Offline fixture follows the community branch contract. The remote arm below
# executes the original Git function itself, not this fixture.
_SOURCE = """def forward(
    self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
    input_ids=None, input_ids_global=None, skip_shared_experts=False,
    return_moe_output=False,
):
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe
    if should_use_mega_moe(self, hidden_states):
        return forward_mega_moe(self, hidden_states, forward_batch,
                                input_ids_global=input_ids_global)
    num_token_non_padded = (forward_batch.moe_num_token_non_padded()
                           if forward_batch is not None else None)
    use_vision_topk = self.gate.e_score_correction_bias_vl is not None
    if not self._enable_a2a_moe:
        if self._can_dual_stream_graph(hidden_states):
            fwd = get_forward()
            return dsv2_flashinfer_moe_dual_stream_graph(
                hidden_states, self.layer_id, fwd.mlp_reduce_scatter)
        elif (self.alt_stream is not None and self.num_fused_shared_experts == 0
              and hidden_states.shape[0] > 0 and get_is_capture_mode()
              and not is_in_breakable_cuda_graph()):
            return self.forward_normal_dual_stream(
                hidden_states, gemm_output_zero_allocator, input_ids,
                input_ids_global=input_ids_global, num_token_non_padded=num_token_non_padded,
                use_vision_topk=use_vision_topk, return_moe_output=return_moe_output)
        return self.forward_normal(
            hidden_states, gemm_output_zero_allocator, input_ids,
            input_ids_global=input_ids_global, skip_shared_experts=skip_shared_experts,
            num_token_non_padded=num_token_non_padded, use_vision_topk=use_vision_topk,
            return_moe_output=return_moe_output)
    return self.forward_deepep(hidden_states, forward_batch, input_ids_global=input_ids_global)
"""


def _record(name):
    def call(*args, **kwargs):
        return name, args, kwargs
    return call


@pytest.fixture(params=["offline", "community-git"])
def moe(request, monkeypatch):
    source = _SOURCE
    if request.param == "community-git":
        tree = os.environ.get("HCU_TEST_SGLANG_TREE")
        if not tree:
            pytest.skip("requires explicitly selected community Git tree")
        tree = Path(tree).resolve()
        text = subprocess.check_output([
            "git", "-c", f"safe.directory={tree}", "-C", str(tree), "show",
            "HEAD:python/sglang/srt/models/deepseek_v2.py",
        ]).decode()
        cls = next(n for n in ast.parse(text).body
                   if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MoE")
        method = next(n for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == "forward")
        source = ast.get_source_segment(text, method)
    state = {"capture": True, "breakable": False, "mega": False, "special": False}
    package = types.ModuleType("sglang.srt.layers.moe.mega_moe")
    package.forward_mega_moe = _record("mega")
    package.should_use_mega_moe = lambda *a: state["mega"]
    # The method has a call-time import; keep the real host out of CPU tests.
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    namespace = {
        "get_is_capture_mode": lambda: state["capture"],
        "is_in_breakable_cuda_graph": lambda: state["breakable"],
        "get_forward": lambda: types.SimpleNamespace(mlp_reduce_scatter=True),
        "dsv2_flashinfer_moe_dual_stream_graph": _record("special"),
        "_is_hip": False,  # CUDA platform semantics; device truth is separate
    }
    exec(compile("from __future__ import annotations\n" + source,
                 "<community-moe-forward>", "exec"), namespace)
    model = type("MoE", (), {
        "forward": namespace["forward"],
        "forward_normal": staticmethod(_record("normal")),
        "forward_normal_dual_stream": staticmethod(_record("dual")),
        "forward_deepep": staticmethod(_record("deepep")),
        "_can_dual_stream_graph": lambda self, x: state["special"],
    })()
    model.alt_stream = object()
    model.num_fused_shared_experts = 0
    model._enable_a2a_moe = False
    model.layer_id = 7
    model.gate = types.SimpleNamespace(e_score_correction_bias_vl=None)
    return model, namespace, state


def test_capture_gate_preserves_alias_and_all_forward_arguments(moe):
    model, namespace, state = moe
    hidden = types.SimpleNamespace(shape=(3, 128))
    alias = model.forward
    method = type(model).forward
    signature = inspect.signature(method)
    assert alias(hidden)[0] == "dual"  # exercise the historical broken branch
    stream = model.alt_stream
    gate._install_forward(method)
    batch = types.SimpleNamespace(moe_num_token_non_padded=lambda: 3)
    result = alias(hidden, batch, "allocator", "ids", input_ids_global="global-ids",
                   skip_shared_experts=True, return_moe_output=True)
    name, args, kwargs = result
    assert name == "normal" and args == (hidden, "allocator", "ids")
    assert kwargs == {
        "input_ids_global": "global-ids", "skip_shared_experts": True,
        "num_token_non_padded": 3, "use_vision_topk": False, "return_moe_output": True,
    }
    assert inspect.signature(method) == signature
    assert namespace["get_is_capture_mode"]() is True  # real graph is still capturing
    assert model.alt_stream is stream  # no temporary shared instance mutation
    assert gate._install_forward(method) is method
    with pytest.raises(TypeError):
        alias(hidden, unexpected=True)


@pytest.mark.parametrize("mode", ["eager", "breakable", "deepep", "mega", "special", "empty"])
def test_other_dispatch_paths_are_unchanged(moe, mode):
    model, _, state = moe
    if mode == "eager":
        state["capture"] = False
    elif mode == "breakable":
        state["breakable"] = True
    elif mode == "deepep":
        model._enable_a2a_moe = True
    elif mode in ("mega", "special"):
        state[mode] = True
    hidden = types.SimpleNamespace(shape=(0 if mode == "empty" else 2, 128))
    before = model.forward(hidden, input_ids_global="global-ids", return_moe_output=True)
    gate._install_forward(type(model).forward)
    assert model.forward(hidden, input_ids_global="global-ids", return_moe_output=True) == before


def test_changed_capture_contract_is_rejected():
    def foreign(self, hidden):
        return hidden
    code = foreign.__code__
    with pytest.raises(RuntimeError, match="capture dispatch changed"):
        gate._install_forward(foreign)
    assert foreign.__code__ is code


def test_public_hip_install_without_source_files(moe, monkeypatch, tmp_path):
    from hcu_adapter import registry

    model, _, _ = moe
    package = types.ModuleType("sglang.srt.models")
    package.__path__ = []
    module = types.ModuleType("sglang.srt.models.deepseek_v2")
    module.__file__ = str(tmp_path / "nonexistent-community-model.py")
    module.DeepseekV2MoE = type(model)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(version=types.SimpleNamespace(hip="6.3")))
    records = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: records.append((a, k)))
    gate.install()
    gate.install()
    assert model.forward(types.SimpleNamespace(shape=(1, 128)))[0] == "normal"
    assert len(records) == 2 and all(
        kwargs["mechanism"] == registry.MECH_REBIND for _, kwargs in records
    )
    assert not list(tmp_path.iterdir())


def test_cuda_host_is_untouched(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(version=types.SimpleNamespace(hip=None)))
    # No sglang import or file lookup is needed on a genuine CUDA host.
    gate.install()
