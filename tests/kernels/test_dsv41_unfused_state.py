"""Original unfused C2 partner selection and pooling with a logical CPU history."""

from types import SimpleNamespace

import pytest
import torch


def _batch(phase, head_dim):
    schedules = (
        [(1, p) for p in range(11)] + [(3, p) for p in range(5)],
        [(3, 5), (1, 11)],
        [(1, p) for p in range(12, 18)] + [(3, p) for p in range(6, 11)],
        [(1, 18), (3, 11)],
    )
    live = schedules[phase]
    # Padding deliberately names an active request. Its positions must not
    # overwrite that request's pending state.
    rows = live + [(1, 100 + i) for i in range(32 - len(live))]
    req = torch.tensor([r for r, _ in rows], dtype=torch.int64)
    pos = torch.tensor([p for _, p in rows], dtype=torch.int64)
    pad = torch.arange(32) >= len(live)
    columns = torch.arange(head_dim)[None, :]
    kv = ((columns * 13 + pos[:, None] * 7 + req[:, None] * 11
           + phase * 17) % 97 - 48).float() / 64
    score = ((columns * 3 + pos[:, None] * 5 + req[:, None] * 7
              + phase * 11) % 19 - 9).float() / 8
    return kv, score, req, pos, pad


def _logical_step(history, batch):
    """Retain every logical token, independent of the device's finite ring."""
    kv, score, req, pos, pad = batch
    partners_kv = torch.zeros_like(kv)
    partners_score = torch.full_like(score, -torch.inf)
    pooled = []
    for row in range(len(req)):
        if not pad[row]:
            key = (int(req[row]), int(pos[row]) - 1)
            if key in history:
                partners_kv[row], partners_score[row] = history[key]
            history[(int(req[row]), int(pos[row]))] = (
                kv[row].clone(), score[row].clone())
        pair = torch.stack((partners_kv[row], kv[row])).double()
        logits = torch.stack((partners_score[row], score[row])).double()
        pooled.append((pair * logits.softmax(dim=0)).sum(dim=0))
    return partners_kv, partners_score, torch.stack(pooled)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="Requires a real HIP device",
)
@pytest.mark.parametrize("ring_size", [4, 8])
@pytest.mark.parametrize("head_dim", [32, 512])
def test_unfused_pair_state_and_fresh_input_graph(ring_size, head_dim):
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
    from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

    state = CompressStatePool(
        size=4 * ring_size, ring_size=ring_size, overlap=False,
        head_dim=head_dim, dtype=torch.float32, device="cuda",
        enable_memory_saver=False, ratio=2, request_scoped=True,
    )
    # Invoke the original backend method and shared state pool. A complete
    # backend/server constructor is unnecessary for this state-only contract.
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_attention_compress_states=lambda layer_id: state)
    batches = [_batch(phase, head_dim) for phase in range(4)]
    buffers = tuple(value.cuda() for value in batches[0])

    def execute():
        kv, score, req, pos, pad = buffers
        partner_kv, partner_score = backend._low_ratio_pair_partners(
            layer_id=0, kv=kv, score=score, req=req, pos=pos, pad=pad)
        pooled = DeepseekV41Compressor.pool_pairs(
            torch.stack((partner_kv, kv), dim=1),
            torch.stack((partner_score, score), dim=1))
        return partner_kv, partner_score, pooled

    # Warm up allocations and capture once. Reset afterward so both execution
    # modes start from the same empty logical history.
    execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute()
    torch.cuda.synchronize()

    for mode in ("eager", "graph"):
        state.kv_score_buffer.clear()
        history = {}
        for batch in batches:
            for buffer, value in zip(buffers, batch):
                buffer.copy_(value)
            if mode == "graph":
                graph.replay()
                actual = captured
            else:
                actual = execute()
            actual = tuple(value.cpu() for value in actual)
            expected = _logical_step(history, batch)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                actual[2].double(), expected[2], rtol=2e-6, atol=2e-6)

            # Derive the finite ring from the unbounded logical history,
            # including untouched request slots and the empty sentinel row.
            stored = state.kv_score_buffer.kv_score.cpu()
            expected_state = torch.zeros_like(stored)
            expected_state[:, head_dim:] = -torch.inf
            for (request, position), (kv, score) in history.items():
                slot = request * ring_size + position % ring_size
                expected_state[slot, :head_dim] = kv
                expected_state[slot, head_dim:] = score
            torch.testing.assert_close(stored, expected_state, rtol=0, atol=0)
