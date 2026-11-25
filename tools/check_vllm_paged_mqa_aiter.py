# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run from the repository root on a gfx936/gfx938 HCU with vLLM installed."""

import json
import runpy

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

    aiter.paged_mqa_logits = counted
    try:
        with torch.inference_mode():
            for pitched in (False, True):
                query, cache, weights, lengths, table, *_ = inputs(
                    device="cuda", padded=pitched, heads=32,
                )
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
                assert len(calls) == 2
                print(json.dumps({
                    "pitched": pitched,
                    "native_calls": len(calls),
                    "graph_replays": 3,
                    "passed": True,
                }), flush=True)
    finally:
        aiter.paged_mqa_logits = native


if __name__ == "__main__":
    main()
