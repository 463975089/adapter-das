import pytest
import torch


requires_hcu = pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")


def assert_logits(actual, expected):
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    valid = torch.isfinite(expected)
    x, y = actual[valid].float(), expected[valid].float()
    if x.numel():
        cosine = torch.nn.functional.cosine_similarity(x, y, dim=0)
        relative = (x - y).abs().max() / y.abs().max().clamp_min(1e-6)
        assert cosine > 0.9999
        assert relative < 0.025


@requires_hcu
@pytest.mark.parametrize("rows,keys", [(1, 129), (37, 513), (128, 4097), (992, 33792)])
def test_public_qsa_prefill_vendor_matches_ragged_reference(rows, keys, monkeypatch):
    from hcu_adapter.qsa_pool import install
    from sglang.srt.layers.attention.qsa import mqa, qsa_indexer
    install()
    torch.manual_seed(rows + keys)
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, 1, 128, device="cuda", dtype=torch.bfloat16)
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 11
    ends = (starts + 1 + torch.arange(rows, device="cuda") * 79).clamp_max(keys).int()
    expected = mqa.torch_qsa_mqa_prefill(q, k, starts, ends, 7.25)
    monkeypatch.setattr(mqa, "torch_qsa_mqa_prefill", lambda *_a, **_k: pytest.fail("vendor route used Torch reference"))
    actual = qsa_indexer.qsa_mqa_prefill(q, k, starts, ends, 7.25)
    assert_logits(actual, expected)


@requires_hcu
@pytest.mark.parametrize("batch,length", [(1, 137), (4, 513), (16, 4097), (4, 32769)])
def test_public_qsa_decode_vendor_graph_uses_live_data(batch, length, monkeypatch):
    from hcu_adapter.qsa_pool import install
    from sglang.srt.layers.attention.qsa import mqa, qsa_indexer
    install()
    torch.manual_seed(batch + length)
    page_size = 16
    pages_per_row = (length + page_size - 1) // page_size
    q = torch.randn(batch, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages_per_row, page_size, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(batch * pages_per_row, device="cuda").reshape(batch, pages_per_row).int()
    lengths = torch.linspace(1, length, batch, device="cuda").int()
    lengths[-1] = length
    reference = mqa.torch_qsa_mqa_decode
    monkeypatch.setattr(mqa, "torch_qsa_mqa_decode", lambda *_a, **_k: pytest.fail("vendor route used Torch reference"))
    call = lambda: qsa_indexer.qsa_mqa_decode(q, k, table, lengths, length, 9.75, min_scored_len=512)
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for i in range(3):
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn_like(k))
        table.copy_(torch.roll(table, 1, dims=1))
        lengths.copy_(torch.linspace(1 + i, length - i, batch, device="cuda").int())
        lengths[-1] = length - i
        if i == 0:
            lengths.zero_()
        graph.replay()
        assert_logits(output, reference(q, k, table, lengths, length, 9.75))


@requires_hcu
def test_qsa_vendor_kill_switch_preserves_torch_reference(monkeypatch):
    from hcu_adapter.qsa_pool import install
    from sglang.srt.layers.attention.qsa import mqa, qsa_indexer
    install()
    monkeypatch.setenv("HCU_DISABLE_OPS", "qsa_mqa")
    q = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(7, 1, 128, device="cuda", dtype=torch.bfloat16)
    starts = torch.zeros(3, device="cuda", dtype=torch.int32)
    ends = torch.tensor([1, 3, 7], device="cuda", dtype=torch.int32)
    expected = mqa.torch_qsa_mqa_prefill(q, k, starts, ends)
    torch.testing.assert_close(qsa_indexer.qsa_mqa_prefill(q, k, starts, ends), expected)
