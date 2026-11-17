# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Non-MX paged MQA: independent math, page-tail layout and replay contracts."""

import pytest
import torch

from hcu_adapter.kernels import mla_attention_emu as emu


def inputs(device="cpu", padded=False, heads=4):
    generator = torch.Generator().manual_seed(33)
    keys = torch.randint(-4, 5, (3, 64, 128), generator=generator).float()
    scales = torch.exp2((torch.arange(3 * 64).reshape(3, 64) % 5 - 2).float())
    # Independent writer: payload for a whole page, THEN FP32 scales.
    storage = torch.zeros(3, 64 * 132 + (256 if padded else 0), dtype=torch.uint8)
    offset = 128 if padded else 0
    payload = storage[:, offset:offset + 64 * 132]
    payload[:, :64 * 128] = keys.to(torch.float8_e4m3fn).view(torch.uint8).reshape(3, -1)
    payload[:, 64 * 128:] = scales.view(torch.uint8).reshape(3, -1)
    gpu_storage = storage.to(device)
    cache = gpu_storage[:, offset:offset + 64 * 132].view(3, 64, 1, 132)
    q = torch.randint(-3, 4, (5, 1, heads, 128), generator=generator).float().to(torch.float8_e4m3fn).to(device)
    weights = torch.randn(5, heads, generator=generator).to(device)
    lengths = torch.tensor([[0], [1], [64], [65], [128]], dtype=torch.int32, device=device)
    table = torch.tensor([[2, 1], [1, 2], [2, 0], [0, 2], [2, 1]], dtype=torch.int32, device=device)
    return q, cache, weights, lengths, table, keys.to(device), scales.to(device)


def expected(q, weights, lengths, table, keys, scales, width):
    # No cache unpacking or production gather helper is used by the reference.
    result = torch.zeros(q.shape[0], width, device=q.device)
    for row in range(q.shape[0]):
        for col in range(min(int(lengths[row, 0]), width)):
            page, slot = int(table[row, col // 64]), col % 64
            scores = (q[row, 0].float() * keys[page, slot]).sum(-1)
            result[row, col] = (scores.relu() * weights[row]).sum() * scales[page, slot]
    return result


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires HCU/CUDA numerical runtime")
    return request.param


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("width", [0, 63, 129])
@pytest.mark.parametrize("indices", [False, True])
def test_non_mx_fp32_page_tail(padded, width, indices, monkeypatch, device):
    q, cache, w, lens, table, keys, scales = inputs(device=device, padded=padded)
    # Force several chunks, including a partial final chunk.
    monkeypatch.setattr(emu, "_LOGITS_CHUNK_ELEMENTS", 5 * 128 * 7)
    owner = torch.arange(5, dtype=torch.int32, device=device) if indices else None
    meta = emu.get_paged_mqa_logits_metadata(lens, 64, 2, owner)
    actual = emu.fp8_fp4_paged_mqa_logits((q, None), cache, w, lens, table, meta, width, owner)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected(q, w, lens, table, keys, scales, width), rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("heads", [32, 64])
def test_model_head_padding_geometry(device, heads):
    q, cache, w, lens, table, keys, scales = inputs(device=device, padded=True, heads=heads)
    # Match the producer's zero-padded scoring-head values/weights.
    q.view(torch.uint8)[:, :, -1] = 0
    w[:, -1] = 0
    actual = emu.fp8_fp4_paged_mqa_logits((q, None), cache, w, lens, table, None, 129)
    torch.testing.assert_close(actual, expected(q, w, lens, table, keys, scales, 129), rtol=1e-4, atol=2e-4)


def test_noncontiguous_block_table_and_empty_batch():
    q, cache, w, lens, table, keys, scales = inputs()
    backing = torch.zeros(5, 4, dtype=torch.int32)
    backing[:, ::2] = table
    actual = emu.fp8_fp4_paged_mqa_logits((q, None), cache, w, lens, backing[:, ::2], None, 129)
    torch.testing.assert_close(actual, expected(q, w, lens, table, keys, scales, 129), rtol=1e-5, atol=2e-5)
    result = emu.fp8_fp4_paged_mqa_logits((q[:0], None), cache, w[:0], lens[:0], table[:0], None, 129)
    assert result.shape == (0, 129)


@pytest.mark.parametrize("bad", ["next_n", "dimension", "cache", "weights", "lengths"])
def test_non_mx_refuses_unsupported_contracts(bad):
    q, cache, w, lens, table, _, _ = inputs()
    if bad == "next_n":
        q = q.expand(-1, 2, -1, -1)
    elif bad == "dimension":
        q = q[..., :64]
    elif bad == "cache":
        cache = cache[..., :131]
    elif bad == "weights":
        w = w[:, :3]
    else:
        lens = lens[:3]
    with pytest.raises((ValueError, NotImplementedError), match="MQA"):
        emu.fp8_fp4_paged_mqa_logits((q, None), cache, w, lens, table, None, 129)


def test_mx_contract_keeps_bfloat16_output():
    q, cache, w, lens, table, _, _ = inputs()
    # Four UE8M0 bytes encode unit scales; a separate writer tests the old MX path.
    packed = torch.full((3, 64), 0x7f7f7f7f, dtype=torch.int32)
    cache.view(3, -1)[:, 64 * 128:] = packed.view(torch.uint8).reshape(3, -1)
    q_sf = torch.full(q.shape[:-1], 0x7f7f7f7f, dtype=torch.int32)
    actual = emu.fp8_fp4_paged_mqa_logits((q, q_sf), cache, w.bfloat16(), lens, table, None, 129, torch.arange(5, dtype=torch.int32))
    assert actual.dtype == torch.bfloat16
    unit = torch.ones(3, 64)
    keys = cache.view(3, -1)[:, :64 * 128].contiguous().view(torch.float8_e4m3fn).reshape(3, 64, 128).float()
    torch.testing.assert_close(actual.float(), expected(q, w.bfloat16().float(), lens, table, keys, unit, 129).bfloat16().float(), rtol=1e-5, atol=1e-5)


def test_vllm_bridge_abi_registration_and_kill_switch(monkeypatch):
    from hcu_adapter import registry
    from hcu_adapter.bridges import vllm_paged_mqa as bridge
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    monkeypatch.setattr(registry, "_RECORDS", dict(registry._RECORDS))
    monkeypatch.setattr(registry, "_BOUND", dict(registry._BOUND))
    metadata, score = bridge.get_paged_mqa_provider()
    q, cache, w, lens, table, keys, scales = inputs()
    result = score((q, None), cache, w, lens, table, metadata(lens, 64, 2), max_model_len=129)
    torch.testing.assert_close(result, expected(q, w, lens, table, keys, scales, 129), rtol=1e-5, atol=2e-5)
    assert registry.get(bridge.OP)["mechanism"] == registry.MECH_SEAM
    assert bridge.get_paged_mqa_provider() == (metadata, score)
    monkeypatch.setenv("HCU_DISABLE_OPS", bridge.OP)
    with pytest.raises(NotImplementedError, match="disabled"):
        bridge.get_paged_mqa_provider()


def test_fresh_process_provider_selection_stays_light():
    import os
    from pathlib import Path
    import subprocess
    import sys

    env = dict(os.environ, HCU_ADAPTER_HOST="none", _HCU_ADAPTER_ACTIVE_HOST="none",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", HCU_DISABLE_OPS="")
    code = """
import sys
from hcu_adapter.bridges.vllm_paged_mqa import (
    fp8_fp4_paged_mqa_logits as existing_alias,
    get_paged_mqa_provider,
)
metadata, score = get_paged_mqa_provider()
assert score is existing_alias
assert get_paged_mqa_provider() == (metadata, score)
assert not any(name == 'torch' or name.startswith(('torch.', 'vllm.')) for name in sys.modules)
"""
    # Disable sitecustomize to measure the explicit bridge, not host bootstrap.
    result = subprocess.run([sys.executable, "-S", "-c", code], env=env,
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA graph runtime")
def test_changing_input_graph_replay():
    q, cache, w, lens, table, keys, scales = inputs(device="cuda", padded=True)
    def run():
        return emu.fp8_fp4_paged_mqa_logits((q, None), cache, w, lens, table, None, 129)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for step in range(2):
        q.copy_(torch.full(q.shape, step + 1, device=q.device).to(q.dtype))
        w.fill_(0.25 * (step + 1))
        lens.copy_(torch.tensor([[1], [0], [65], [64], [127]], dtype=torch.int32, device=q.device))
        table.copy_(table.roll(1, 0))
        keys.neg_()
        scales.mul_(0.5)
        payload = cache.as_strided((3, 64 * 132), (cache.stride(0), 1))
        payload[:, :64 * 128].copy_(keys.to(torch.float8_e4m3fn).view(torch.uint8).reshape(3, -1))
        payload[:, 64 * 128:].copy_(scales.view(torch.uint8).reshape(3, -1))
        graph.replay()
        torch.testing.assert_close(actual, expected(q, w, lens, table, keys, scales, 129), rtol=1e-4, atol=1e-4)
