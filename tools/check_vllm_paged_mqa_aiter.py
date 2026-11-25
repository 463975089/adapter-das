# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run from the repository root on a gfx936/gfx938 HCU with vLLM installed."""

import json
import runpy
from itertools import product

import aiter
import torch

from hcu_adapter.kernels.mla_attention_emu import (
    fp8_fp4_paged_mqa_logits as reference,
)
from hcu_adapter.kernels.vllm_paged_mqa_aiter import (
    fp32_paged_mqa_logits as candidate,
)
from vllm.v1.worker.workspace import init_workspace_manager


def main():
    inputs = runpy.run_path("tests/kernels/test_vllm_paged_mqa_fp32.py")["inputs"]
    torch.cuda.set_device(0)
    init_workspace_manager(torch.device("cuda:0"))
    calls = []
    native = aiter.paged_mqa_logits

    def counted(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    supports_stride = (
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx938"
        and getattr(native, "supports_page_stride", False) is True
    )
    counted.supports_page_stride = getattr(native, "supports_page_stride", False)
    aiter.paged_mqa_logits = counted
    expected_calls = 0
    try:
        with torch.inference_mode():
            for heads, page_stride in product((32, 64), (8448, 8704, 2015232)):
                pitched = page_stride != 8448
                query, cache, weights, lengths, table, *_ = inputs(
                    device="cuda", padded=False, heads=heads,
                )
                if pitched:
                    storage = torch.full(
                        (cache.shape[0], page_stride), 197,
                        device="cuda", dtype=torch.uint8,
                    )
                    payload = storage[:, 128:128 + 8448]
                    payload.copy_(cache.reshape(cache.shape[0], 8448))
                    cache = payload.view(cache.shape[0], 64, 1, 132)
                table[0].fill_(-1)

                def run():
                    return candidate(
                        (query, None), cache, weights, lengths, table, None, 129,
                    )

                output = run()
                expected = reference(
                    (query, None), cache, weights, lengths, table, None, 129,
                )
                torch.testing.assert_close(output, expected, atol=0.002, rtol=0.002)
                saved = output.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                for _ in range(3):
                    query.copy_(query.flip(0))
                    weights.mul_(-0.8)
                    lengths.copy_(lengths.roll(1, 0))
                    table.copy_(table.roll(1, 0))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = reference(
                        (query, None), cache, weights, lengths, table, None, 129,
                    )
                    torch.testing.assert_close(captured, expected, atol=0.002, rtol=0.002)
                    valid = torch.arange(129, device="cuda")[None, :] < lengths
                    assert torch.equal(captured[~valid], torch.zeros_like(captured[~valid]))
                torch.testing.assert_close(output, saved, atol=0, rtol=0)
                if not pitched or supports_stride:
                    expected_calls += 2
                assert len(calls) == expected_calls
                print(json.dumps({
                    "pitched": pitched,
                    "heads": heads,
                    "page_stride": page_stride,
                    "native_calls": len(calls),
                    "graph_replays": 3,
                    "passed": True,
                }), flush=True)
    finally:
        aiter.paged_mqa_logits = native


if __name__ == "__main__":
    main()
