"""Byte-plane copy contract, including fresh device metadata on graph replay."""
import pytest
import torch

from hcu_adapter.kernels.fp4_staging_triton import (
    stage_decode_pages, supports_staging_device,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and supports_staging_device('cuda')),
    reason='FP4 staging requires HIP gfx938 with 64 CUs',
)


def check_cpu_oracle(cache, table, lengths, pages, out_table):
    cache, table, lengths = cache.cpu(), table.cpu(), lengths.cpu()
    pages, out_table = pages.cpu(), out_table.cpu()
    records = cache.shape[1]
    raw = cache.reshape(cache.shape[0], records * 68)
    max_pages = pages.shape[1]
    for row in range(len(lengths)):
        length = min(int(lengths[row]), max_pages * 64)
        live_pages = (length + 63) // 64
        for page in range(max_pages):
            if page >= live_pages:
                assert (pages[row, page] == 137).all()
                assert int(out_table[row, page]) == -37
                continue
            # Exact independent page-local payload/scale-plane addresses.
            pos = page * 64 + torch.arange(64)
            physical = table[row, pos // records].long()
            slot = pos % records
            physical[pos >= length] = 0
            slot[pos >= length] = 0
            payload = raw[physical[:, None], slot[:, None] * 64 + torch.arange(64)]
            scale = raw[physical[:, None], records * 64 + slot[:, None] * 4 + torch.arange(4)]
            expected = torch.cat((payload.flatten(), scale.flatten()))
            assert torch.equal(pages[row, page], expected)
            assert int(out_table[row, page]) == row * max_pages + page


@pytest.mark.parametrize('records', [32, 128])
@pytest.mark.parametrize('rows,capacity', [(0, 0), (1, 65), (7, 513), (17, 1025), (42, 4097), (129, 257), (257, 65)])
def test_copy_and_changed_metadata_graph(records, rows, capacity):
    torch.manual_seed(records + rows)
    max_pages = (capacity + 63) // 64
    cols = (max_pages * 64 + records - 1) // records
    cache = torch.randint(0, 256, (1024, records, 68), device='cuda', dtype=torch.uint8)
    storage = torch.randint(0, 1024, (rows, cols * 2), device='cuda', dtype=torch.int32)
    table = storage[:, ::2]
    lengths = torch.randint(0, capacity + 1, (rows,), device='cuda', dtype=torch.int32)
    pages = torch.full((rows, max_pages, 4352), 137, device='cuda', dtype=torch.uint8)
    out_storage = torch.full((rows, max_pages * 2), -37, device='cuda', dtype=torch.int32)
    out_table = out_storage[:, ::2]

    def execute():
        assert stage_decode_pages(cache, table, lengths, pages, out_table, live_only=True)

    execute()
    torch.cuda.synchronize()
    check_cpu_oracle(cache, table, lengths, pages, out_table)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        execute()
    for replay in range(3):
        lengths.random_(0, capacity + 1)
        storage.random_(0, 1024)
        pages.fill_(137)
        out_table.fill_(-37)
        graph.replay()
        torch.cuda.synchronize()
        check_cpu_oracle(cache, table, lengths, pages, out_table)
        assert (out_storage[:, 1::2] == -37).all()
