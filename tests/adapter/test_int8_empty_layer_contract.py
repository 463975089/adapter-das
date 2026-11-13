# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real CT parse, Linear constructors/loaders and INT8 postprocess on CPU.

Only hardware admission and nonempty quantization/GEMM use explicit doubles.
Empty inputs must never reach either compute double. No plugin is required.
"""
import pytest

import torch
from hcu_adapter import dsv41_int8_linear as owned
from vllm import _custom_ops as ops
from vllm.model_executor import parameter
from vllm.model_executor.kernels import linear as kernel_factory
from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterInt8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonInt8ScaledMMLinearKernel
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization import base_config
from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors as ct
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as int8,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)

K, N = 16, 6
CALLS = []
KERNEL = [TritonInt8ScaledMMLinearKernel]


def ct_description(strategy="channel", static=False, symmetric=True):
    """A real compressed-tensors checkpoint schema, not a hand-built scheme."""
    weight = {"num_bits": 8, "type": "int", "symmetric": True, "dynamic": False, "strategy": strategy}
    act = {
        "num_bits": 8, "type": "int", "symmetric": symmetric, "dynamic": not static,
        "strategy": "tensor" if static else "token",
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "linear": {"targets": ["Linear"], "format": "int-quantized", "weights": weight, "input_activations": act},
        },
        "ignore": [r"re:.*\.ignored$"],
    }


class _OwnedCT(ct.CompressedTensorsConfig):
    """Real CT parse and scheme construction; only the INT8 selector is ours.

    Hardware availability is doubled; the real CT method owns weight creation,
    loading and postprocess for every supported Linear class.
    """

    def get_quant_method(self, layer, prefix):
        method = super().get_quant_method(layer, prefix)
        if (
            isinstance(layer, linear.LinearBase)
            and type(method) is CompressedTensorsLinearMethod
            and type(getattr(layer, "scheme", None)) is CompressedTensorsW8A8Int8
        ):
            return owned.HcuDSV41Int8LinearMethod(self)
        return method


def config(**kwargs):
    return _OwnedCT.from_config(ct_description(**kwargs))


def select_kernel(**kwargs):
    layer_config = kernel_factory.Int8ScaledMMLinearLayerConfig(
        is_channelwise=kwargs["is_channelwise"],
        is_static_input_scheme=kwargs["is_static_input_scheme"],
        input_symmetric=kwargs["input_symmetric"],
        is_fused_module=kwargs.get("is_fused_module", False),
    )
    return KERNEL[0](layer_config, ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"])


def quantize(x, scale=None, zp=None, symmetric=True):
    rows = x.numel() // x.shape[-1]
    if rows == 0:
        raise RuntimeError("empty input reached native quantization")
    CALLS.append("quant")
    data = x.reshape(rows, x.shape[-1]).to(torch.int8)
    scales = torch.ones((rows, 1), dtype=torch.float32) if scale is None else scale
    zero = None if symmetric else (torch.zeros((rows, 1), dtype=torch.int32) if zp is None else zp)
    return data, scales, zero


def gemm(x, w, xs, ws, out_dtype, bias=None):
    if x.shape[0] == 0:
        raise RuntimeError("empty input reached native GEMM")
    CALLS.append("gemm")
    out = (x.float() @ w.float()) * xs * ws.reshape(1, -1)
    return (out if bias is None else out + bias).to(out_dtype)


def build(kind="column", cfg=None, k=K, n=N, finalize=True, **kwargs):
    cfg = config() if cfg is None else cfg
    common = dict(
        quant_config=cfg, params_dtype=torch.bfloat16, bias=False,
        prefix="model.layers.0.wkv", disable_tp=True,
    )
    common.update(kwargs)
    if kind == "merged":
        obj = linear.MergedColumnParallelLinear(k, [n // 2, n - n // 2], **common)
    elif kind == "replicated":
        obj = linear.ReplicatedLinear(k, n, **common)
    elif kind == "row":
        obj = linear.RowParallelLinear(k, n, **common)
    else:
        obj = linear.ColumnParallelLinear(k, n, **common)
    # Real CT layout: the loader hands (N, K) rows and native finalization transposes.
    weight = torch.arange(n * k, dtype=torch.int32).remainder(17).reshape(n, k).to(torch.int8)
    scales = torch.arange(1, n + 1, dtype=torch.float32).reshape(n, 1)
    if kind == "merged":
        for shard, width in enumerate(obj.output_partition_sizes):
            start = sum(obj.output_partition_sizes[:shard])
            obj.weight.weight_loader(obj.weight, weight[start:start + width], shard)
            shard_scale = scales[start:start + width] if obj.scheme.strategy == "channel" else torch.tensor(float(shard + 1))
            obj.weight_scale.weight_loader(obj.weight_scale, shard_scale, shard)
    else:
        obj.weight.weight_loader(obj.weight, weight)
        obj.weight_scale.weight_loader(
            obj.weight_scale, scales if obj.scheme.strategy == "channel" else torch.tensor(2.0)
        )
    if obj.input_scale is not None:
        obj.input_scale.weight_loader(obj.input_scale, torch.tensor([0.5]))
    if obj.input_zero_point is not None:
        obj.input_zero_point.weight_loader(obj.input_zero_point, torch.tensor([1], dtype=torch.int8))
    if finalize:
        obj.quant_method.process_weights_after_loading(obj)
    return obj, weight, scales



@pytest.fixture(autouse=True)
def native_cpu_contract(monkeypatch):
    monkeypatch.setattr(ct.CompressedTensorsConfig, '_check_scheme_supported', lambda *a, **kw: True)
    for cls in (TritonInt8ScaledMMLinearKernel, AiterInt8ScaledMMLinearKernel):
        monkeypatch.setattr(cls, 'is_supported', lambda *a: (True, None))
    monkeypatch.setattr(int8, 'init_int8_linear_kernel', select_kernel)
    for module in (parameter, linear):
        monkeypatch.setattr(module, 'get_tensor_model_parallel_rank', lambda: 0)
        monkeypatch.setattr(module, 'get_tensor_model_parallel_world_size', lambda: 1)
    monkeypatch.setattr(ops, 'scaled_int8_quant', quantize)
    monkeypatch.setattr(torch.ops.vllm, 'int8_scaled_mm_func', gemm)


@pytest.mark.parametrize('kind', ['replicated', 'row', 'column', 'merged'])
@pytest.mark.parametrize('shape', [(0, K), (2, 0, K), (0, 3, K)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_empty_real_linear_layers(kind, shape, dtype):
    obj, _, _ = build(kind)
    if kind == 'replicated':
        assert not hasattr(obj, 'input_size_per_partition')
        assert not hasattr(obj, 'output_size_per_partition')
    before = len(CALLS)
    out, _ = obj(torch.empty(shape, dtype=dtype))
    assert out.shape == (*shape[:-1], N)
    assert out.dtype == dtype and out.device.type == 'cpu'
    assert len(CALLS) == before


@pytest.mark.parametrize('kind', ['replicated', 'row', 'column', 'merged'])
@pytest.mark.parametrize('dimension', ['input', 'output'])
def test_mismatched_layer_dimensions_still_rejected(kind, dimension, monkeypatch):
    obj, _, _ = build(kind)
    attr = dimension + ('_size' if kind == 'replicated' else '_size_per_partition')
    monkeypatch.setattr(obj, attr, getattr(obj, attr) + 1)
    with pytest.raises(ValueError, match='dimensions must match local partitions'):
        obj(torch.empty((0, K), dtype=torch.bfloat16))


def test_nonempty_replicated_still_delegates_native():
    obj, weight, scales = build('replicated')
    x = torch.ones((2, K), dtype=torch.bfloat16)
    before = len(CALLS)
    out, _ = obj(x)
    expected = ((x.float() @ weight.float().t()) * scales.t()).to(x.dtype)
    assert torch.equal(out, expected)
    assert CALLS[before:] == ['quant', 'gemm']
