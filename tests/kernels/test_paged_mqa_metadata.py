"""get_paged_mqa_logits_metadata contract (DSA indexer paged-MQA trio).

The DTK deepgemm wheel lacks the trio, so deep_gemm_dtk installs the
dsa_mqa_logits_emu provider. Its blob must satisfy every consumer of the
same symbol:

1. FALLBACK  — without lightop: [rows + 1, 2] int32 zeros on the input
               device (torch/triton tiers ignore it).
2. PLANNER   — with lightop: the lightop split-KV planner's own output,
               fed int32 contiguous 1-D lens (its paged_mqa_logits asserts
               a genuine plan; a zero blob made decode graph capture fail
               on S1 Hy4 2026-11-04). The plan is sized by num_sms, not by
               the row count ([2 * num_sms + 1, 2] on the 2026-11 wheel).
3. REPLAY    — the planner is capture-safe: a captured call re-reads the
               lens tensor on replay and matches an eager call.

Run: pytest tests/kernels/test_paged_mqa_metadata.py -v
"""
import importlib.util

import pytest
import torch

import hcu_adapter.dsa_mqa_logits_emu as M

requires_lightop_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("lightop")),
    reason="requires HCU device + lightop",
)


def _with_planner(fn):
    saved = list(M._LIGHTOP_META)
    M._LIGHTOP_META[:] = [fn]
    return saved


def test_fallback_without_lightop():
    saved = _with_planner(None)
    try:
        meta = M.get_paged_mqa_logits_metadata(
            torch.tensor([[3], [70], [129]], dtype=torch.int32), 64, 80)
    finally:
        M._LIGHTOP_META[:] = saved
    assert meta.shape == (4, 2) and meta.dtype == torch.int32
    assert not meta.any()


def test_planner_receives_flat_int32_lens():
    seen = {}

    def planner(lens, block_kv, num_sms):
        seen.update(lens=lens, block_kv=block_kv, num_sms=num_sms)
        return torch.ones(num_sms + 1, 2, dtype=torch.int32)

    saved = _with_planner(planner)
    try:
        meta = M.get_paged_mqa_logits_metadata(
            torch.tensor([[5], [9]], dtype=torch.int64), 64, 16)
    finally:
        M._LIGHTOP_META[:] = saved
    assert meta.shape == (17, 2)
    assert seen["lens"].dtype == torch.int32 and seen["lens"].dim() == 1
    assert seen["lens"].is_contiguous() and seen["lens"].tolist() == [5, 9]
    assert (seen["block_kv"], seen["num_sms"]) == (64, 16)


@requires_lightop_gpu
def test_lightop_plan_shape():
    M._LIGHTOP_META.clear()
    num_sms = M.get_num_sms()
    lens = torch.randint(1, 5000, (160, 1), dtype=torch.int32, device="cuda")
    meta = M.get_paged_mqa_logits_metadata(lens, 64, num_sms)
    assert meta.dim() == 2 and meta.shape[1] == 2 and meta.dtype == torch.int32
    small = M.get_paged_mqa_logits_metadata(lens[:3], 64, num_sms)
    assert small.shape == meta.shape


@requires_lightop_gpu
def test_lightop_plan_graph_replay():
    M._LIGHTOP_META.clear()
    lens = torch.full((8, 1), 300, dtype=torch.int32, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        M.get_paged_mqa_logits_metadata(lens, 64, 64)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = M.get_paged_mqa_logits_metadata(lens, 64, 64)
    lens.fill_(4000)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, M.get_paged_mqa_logits_metadata(lens, 64, 64))
