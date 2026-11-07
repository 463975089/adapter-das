# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native router binding, score semantics and changed-input graph regression."""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hcu_adapter import moe_topk_order as bridge


# Keep the native routing operations in the fixture, so independent expected
# IDs/weights below can detect a change to grouping, bias or normalization.
_SOURCE = """
def router(hidden_states, gating_output, correction_bias, topk, renormalize,
           num_expert_group=1, topk_group=1, num_fused_shared_experts=0,
           routed_scaling_factor=None, apply_routed_scaling_factor_on_output=False):
    scores = gating_output.sigmoid()
    n, e = scores.shape
    scores_for_choice = scores + correction_bias.view(-1, e)
    group_scores = scores_for_choice.view(n, num_expert_group, -1).topk(2, dim=-1)[0].sum(-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    mask = group_mask.unsqueeze(-1).expand(n, num_expert_group, e // num_expert_group).reshape(n, e)
    choice = scores_for_choice.masked_fill(~mask.bool(), float("-inf"))
    _, topk_ids = torch.topk(choice, k=topk, dim=-1, sorted=(True if num_fused_shared_experts > 0 else False))
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True, dtype=torch.float32) + 1e-20)
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor
    return topk_weights.float(), topk_ids.int()
"""


def _case(source=_SOURCE):
    namespace = {"torch": torch}
    exec(compile(source, "<native-router-fixture>", "exec"), namespace)
    return namespace["router"]


def test_only_validated_sorted_operands_change():
    function = _case()
    original = function.__code__
    adapted = bridge._ordered_code(original)
    assert adapted.co_code == original.co_code
    assert adapted.co_varnames == original.co_varnames
    assert adapted.co_argcount == original.co_argcount
    assert [a for a, b in zip(original.co_consts, adapted.co_consts) if a is not b] == [False]
    source = _SOURCE.replace("scores = gating_output.sigmoid()",
                             "unrelated = False\n    scores = gating_output.sigmoid()")
    with pytest.raises(RuntimeError, match="operands changed"):
        bridge._ordered_code(_case(source).__code__)


def test_aliases_signature_defaults_and_cpu_behavior_survive_install():
    function = _case()
    alias = function
    signature = inspect.signature(function)
    x = torch.arange(72, dtype=torch.float32).reshape(1, 72) / 24
    args = (x, x, torch.zeros(72), 6, True)
    before = function(*args)
    assert bridge._install_function(function) is function is alias
    assert inspect.signature(alias) == signature
    after = alias(*args)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert bridge._install_function(function) is function
    assert function.__globals__["torch"] is torch
    original, adapted, stock, ordered = getattr(function, bridge._MARKER)
    assert ordered.__globals__ is stock.__globals__ is function.__globals__
    assert stock.__code__ is original and ordered.__code__ is adapted
    function.__globals__[bridge._DISPATCH] = object()
    with pytest.raises(RuntimeError, match="binding changed"):
        bridge._install_function(alias)


@pytest.mark.parametrize("source", [
    _SOURCE.replace("topk, renormalize,", "topk, renamed,"),
    _SOURCE.replace("num_expert_group=1", "*, num_expert_group=1"),
    _SOURCE.replace("torch.topk(choice", "torch.sort(choice"),
])
def test_upstream_drift_is_rejected_before_mutation(source):
    function = _case(source)
    code = function.__code__
    with pytest.raises(RuntimeError, match="changed"):
        bridge._install_function(function)
    assert function.__code__ is code
    assert not hasattr(function, bridge._MARKER)
    assert bridge._DISPATCH not in function.__globals__


def test_compiled_and_cached_original_aliases_survive_warm_install():
    function = _case()
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    original_alias = inspect.unwrap(compiled)
    backend_alias = compiled._torchdynamo_orig_callable
    compiled_alias = compiled
    x = torch.arange(72, dtype=torch.float32).reshape(1, 72) / 24
    args = (x, x, torch.zeros(72), 6, True)
    expected = compiled(*args)
    assert bridge._install_function(compiled) is compiled_alias
    assert compiled._torchdynamo_orig_callable is backend_alias
    assert inspect.unwrap(compiled) is original_alias is function
    assert all(torch.equal(a, b) for a, b in zip(compiled_alias(*args), expected))
    assert all(torch.equal(a, b) for a, b in zip(original_alias(*args), expected))
    assert all(torch.equal(a, b) for a, b in zip(backend_alias(*args), expected))


def test_install_disabled_and_non_hip_do_not_import_host(monkeypatch):
    from hcu_adapter import hcu_env

    imported = []
    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("sglang"):
            imported.append(name)
            raise AssertionError("disabled/native CUDA bridge imported SGLang")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(hcu_env, "enabled", lambda name: False)
    bridge.install()
    monkeypatch.setattr(hcu_env, "enabled", lambda name: True)
    monkeypatch.setattr(torch.version, "hip", None)
    bridge.install()
    assert imported == []


def test_fresh_interpreter_binding_is_process_local(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "spawn_check.py"
    source.write_text(
        "import types\n"
        "from hcu_adapter import moe_topk_order as b\n"
        f"scope={{'torch': __import__('torch')}}\nexec({_SOURCE!r}, scope)\n"
        "f=scope['router']; alias=f; b._install_function(f)\n"
        "assert alias is f and f.__code__ is b._native.__code__\n"
        "assert scope['torch'] is __import__('torch')\n",
        encoding="utf-8",
    )
    env = dict(os.environ, HCU_ADAPTER_HOST="none", PYTHONPATH=str(root),
               HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, str(source)], env=env, capture_output=True,
                            text=True, timeout=90)
    assert result.returncode == 0, result.stderr


_HIP = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(),
    reason="requires visible HIP device",
)


@_HIP
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("groups,selected_groups", [(1, 1), (4, 2)])
def test_device_selection_weights_and_repeated_order(dtype, groups, selected_groups):
    function = _case()
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    bridge._install_function(compiled)
    # Wide score gaps give unambiguous selected sets in all three dtypes.
    logits = torch.linspace(-3, 3, 72, device="cuda", dtype=dtype).repeat(17, 1)
    bias = torch.zeros(72, device="cuda")
    outputs = [compiled(logits, logits, bias, 6, True, groups, selected_groups,
                        0, 2.0, True) for _ in range(8)]
    expected_ids = torch.arange(71, 65, -1, device="cuda", dtype=torch.int32).expand(17, -1)
    # Independent CPU sigmoid, with the same declared activation storage dtype.
    scores = torch.sigmoid(logits.cpu().double()).to(dtype).double()[:, -6:].flip(-1)
    expected_weights = (2 * scores / (scores.sum(-1, keepdim=True) + 1e-20)).float()
    for weights, ids in outputs:
        assert torch.equal(ids, expected_ids)
        torch.testing.assert_close(weights.cpu(), expected_weights, rtol=2e-5, atol=2e-6)
        assert torch.equal(weights, outputs[0][0])
        assert torch.equal(ids, outputs[0][1])


@_HIP
@pytest.mark.parametrize("renormalize", [False, True])
def test_bias_selects_experts_without_entering_weights(renormalize):
    function = _case()
    bridge._install_function(function)
    logits = torch.zeros(5, 72, device="cuda")
    bias = torch.linspace(0, 1, 72, device="cuda")
    weights, ids = function(logits, logits, bias, 6, renormalize, 4, 2, 0, 2.0, True)
    expected_ids = torch.arange(71, 65, -1, device="cuda", dtype=torch.int32).expand(5, -1)
    assert torch.equal(ids, expected_ids)
    expected_weight = 2.0 / 6 if renormalize else 0.5
    torch.testing.assert_close(weights, torch.full_like(weights, expected_weight),
                               rtol=1e-6, atol=1e-7)
    # At a genuine tie, upstream does not specify the expert IDs or tie order.
    # Preserve valid distinct experts and unbiased normalized weights.
    tied_weights, tied_ids = function(logits, logits, torch.zeros_like(bias), 6, True)
    assert bool(((tied_ids >= 0) & (tied_ids < 72)).all())
    assert not bool((tied_ids.sort(-1).values[:, 1:] == tied_ids.sort(-1).values[:, :-1]).any())
    torch.testing.assert_close(tied_weights, torch.full_like(tied_weights, 1.0 / 6),
                               rtol=1e-6, atol=1e-7)


@_HIP
def test_changed_input_graph_uses_fresh_routes_and_weights():
    function = _case()
    bridge._install_function(function)
    compiled = torch.compile(function, backend="eager", fullgraph=True)
    x = torch.linspace(-3, 3, 72, device="cuda").repeat(437, 1)
    bias = torch.zeros(72, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(x, x, bias, 6, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(x, x, bias, 6, True)
    for reverse, shift in ((False, 0.0), (True, 0.0), (False, -0.75)):
        fresh = torch.linspace(-3, 3, 72, device="cuda").repeat(437, 1)
        if reverse:
            fresh = fresh.flip(-1)
        fresh.add_(shift)
        x.copy_(fresh)
        expected = compiled(fresh, fresh, bias, 6, True)
        graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(captured, expected))
