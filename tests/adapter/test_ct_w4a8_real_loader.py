"""Real community callback fixture, not model construction or numerical acceptance.

Select HCU_TEST_SGLANG_TREE and a fresh SGLang host process. Tensor fixtures stay
on CPU, but importing the real host requires its normal runtime dependencies.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
import os
import subprocess

import pytest
import torch

from hcu_adapter import ct_w4a8_int8_moe as native, hcu_env


@pytest.fixture
def community(monkeypatch):
    selected = os.environ.get("HCU_TEST_SGLANG_TREE")
    if not selected or hcu_env.resolve("HCU_ADAPTER_HOST")[0] != "sglang":
        pytest.skip("requires selected community Git tree in a fresh SGLang host")
    from compressed_tensors.quantization import QuantizationArgs
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsWNA16TritonMoE,
    )

    tree = Path(selected).resolve()
    source = Path(moe.__file__).resolve()
    relative = "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
    assert source == tree / relative, "wrong host source"
    original = source.read_bytes()
    assert original == subprocess.check_output([
        "git", "-c", f"safe.directory={tree}", "-C", str(tree),
        "show", f"HEAD:{relative}",
    ])
    function = moe.FusedMoE._weight_loader_impl
    code = function.__code__
    adapted = native._adapted_loader_code(function)
    monkeypatch.setattr(function, "__code__", adapted)
    assert adapted.co_code == code.co_code
    args = QuantizationArgs(num_bits=4, type="int", strategy="channel",
                            symmetric=True, dynamic=False, group_size=-1)
    scheme = native._make_scheme_class(CompressedTensorsWNA16TritonMoE)(
        NS(quant_format="pack-quantized"), args,
    )
    try:
        yield moe.FusedMoE, MoeRunnerConfig, scheme, function
    finally:
        assert source.read_bytes() == original


@pytest.mark.parametrize("tp_size,tp_rank,ep_size,ep_rank", [
    (1, 0, 1, 0), (2, 0, 1, 0), (2, 1, 1, 0),
    (1, 0, 2, 1), (2, 1, 2, 1),
])
def test_actual_callback_packed_scales_tp_ep_and_async(
    community, monkeypatch, tp_size, tp_rank, ep_size, ep_rank,
):
    FusedMoE, RunnerConfig, scheme, alias = community
    experts, hidden, intermediate = 2, 32, 32
    local_intermediate = intermediate // tp_size
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    for key, value in dict(
        moe_tp_size=tp_size, moe_tp_rank=tp_rank, num_local_experts=experts,
        _expert_storage_rank=ep_rank, _num_local_routed=experts,
        _num_global_routed=experts * ep_size, _has_fused_shared=False,
        num_fused_shared_experts=0, layer_id=0, quant_config=None,
        use_triton_kernels=False, use_padded_loading=False,
        use_presharded_weights=False, use_flashinfer_trtllm_moe=False,
        moe_runner_config=RunnerConfig(is_gated=True), scheme=scheme,
        quant_method=NS(),
    ).items():
        setattr(layer, key, value)
    # Use the real allocation and public callback, not direct parameter copies.
    with torch.device("cpu"):
        scheme.create_weights(layer, experts, hidden, local_intermediate,
                              torch.bfloat16, weight_loader=layer.weight_loader)
    gate = torch.arange(intermediate * hidden // 8).reshape(intermediate, hidden // 8).int()
    up = gate + 30000
    down = torch.arange(hidden * intermediate // 8).reshape(hidden, intermediate // 8).int()
    gate_scale = torch.arange(1, intermediate + 1).bfloat16()[:, None]
    up_scale = gate_scale + 50
    down_scale = torch.arange(3, hidden + 3).bfloat16()[:, None]
    weights = dict(w1=(gate, gate_scale), w3=(up, up_scale), w2=(down, down_scale))

    # Prove that omitting the candidate from the original membership test is
    # observable through the real callback, before validating the fix.
    adapted = alias.__code__
    stock = adapted.replace(co_consts=tuple(
        native._LOADER_SCHEMES
        if value == native._LOADER_SCHEMES + (native._CANDIDATE_SCHEME,) else value
        for value in adapted.co_consts
    ))
    with monkeypatch.context() as temporary:
        temporary.setattr(alias, "__code__", stock)
        with pytest.raises(RuntimeError):
            layer.w13_weight_scale.weight_loader(
                layer.w13_weight_scale, gate_scale, "w1.weight_scale", "w1",
                ep_rank * experts,
            )
    assert alias is FusedMoE._weight_loader_impl and alias.__code__ is adapted

    calls = []
    for expert in range(experts):
        for tag, tensors in weights.items():
            label = "w2" if tag == "w2" else "w13"
            for kind, tensor in zip(("weight_packed", "weight_scale"), tensors):
                param = getattr(layer, label + "_" + kind)
                calls.append((param, tensor, tag + "." + kind, tag,
                              ep_rank * experts + expert))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args: args[0].weight_loader(*args), calls))
    unchanged = layer.w13_weight_scale.clone()
    layer.w13_weight_scale.weight_loader(
        layer.w13_weight_scale, torch.ones(2, 3), "w1.weight_scale", "w1",
        (1 - ep_rank) * experts if ep_size == 2 else experts,
    )
    assert torch.equal(layer.w13_weight_scale, unchanged)

    begin, end = tp_rank * local_intermediate, (tp_rank + 1) * local_intermediate
    expected_gate_up = torch.cat((gate[begin:end], up[begin:end]))
    expected_down = down[:, begin // 8:end // 8].contiguous()
    expected_scales = torch.cat((gate_scale[begin:end], up_scale[begin:end]))
    scheme.process_weights_after_loading(layer)
    for expert in range(experts):
        assert torch.equal(layer.w13_weight_packed[expert],
                           expected_gate_up.contiguous().view(torch.uint8))
        assert torch.equal(layer.w2_weight_packed[expert], expected_down.view(torch.uint8))
        assert torch.equal(layer.w13_weight_scale[expert], expected_scales)
        assert torch.equal(layer.w2_weight_scale[expert], down_scale)
