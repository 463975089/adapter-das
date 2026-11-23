# SPDX-License-Identifier: Apache-2.0
"""Real HIP registration/mapped-span contracts; no Core or allocator substitutes.

Run with the native HIP lane on one explicitly assigned GPU. Registration is
owned by each fixture until synchronized teardown, including failing subcases.
"""
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(torch.version.hip is None, reason="requires HIP Torch")


@pytest.fixture(scope="module")
def registered_ops(request):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        pytest.skip("requires one explicitly assigned HIP device; not validated")
    from hcu_adapter.kernels.hisparse import ensure_hisparse_ops
    torch.cuda.set_device(request.config.getoption("--hisparse-device"))
    ensure_hisparse_ops()  # discovery of the complete installed native wheel
    return torch.ops._C_cache_ops


@contextmanager
def registration(pages=8, regions=None):
    """Keep CPU storage and the exact HIP registrations live through all work."""
    page = 4096
    owner = torch.empty((pages + 1) * page, dtype=torch.uint8)
    offset = (-owner.data_ptr()) % page
    aligned = owner[offset:offset + pages * page]
    regions = [(0, pages)] if regions is None else regions
    live = []
    rt = torch.cuda.cudart()
    try:
        for first, count in regions:
            ptr = aligned.data_ptr() + first * page
            error = rt.cudaHostRegister(ptr, count * page, 2)  # HostRegisterMapped
            assert error.value == 0, error
            live.append(ptr)
        yield aligned, live
    finally:
        # Never free/unregister storage with in-flight work, even on test error.
        torch.cuda.synchronize()
        while live:
            error = rt.cudaHostUnregister(live[-1])
            assert error.value == 0, error
            live.pop()


@pytest.mark.parametrize("dtype,columns", [(torch.bfloat16, 576), (torch.uint8, 656)])
@pytest.mark.parametrize("block_size", [64, 256])
@pytest.mark.parametrize("kind", ["plan", "compact"])
@pytest.mark.parametrize("offset", [16, 4096])
def test_registered_typed_gather_offsets_and_guards(
        registered_ops, dtype, columns, block_size, kind, offset):
    with registration() as (pool, _):
        rows, blocks, gap, guard = 8, 3, 24, 32
        item = torch.empty((), dtype=dtype).element_size()
        row_bytes = columns * item
        values = (torch.arange(pool.numel(), dtype=torch.int64) * 17 % 251).to(torch.uint8)
        pool.copy_(values)
        host = pool[offset:offset + rows * row_bytes].view(dtype).view(rows, columns)
        pitch = block_size * row_bytes + gap
        size = offset + blocks * pitch + guard
        owner = torch.full((size,), 85, dtype=torch.uint8, device="cuda")
        hot = owner.view(dtype).as_strided((blocks, block_size, columns),
                  (pitch // item, columns, 1), offset // item)
        expected = torch.full((size,), 85, dtype=torch.uint8)
        sources = [0, 3, rows + 1]
        destinations = [block_size - 1, block_size, blocks * block_size - 1]
        for source, target in zip(sources, destinations):
            begin = offset + target // block_size * pitch + target % block_size * row_bytes
            if source < rows:
                src = offset + source * row_bytes
                expected[begin:begin + row_bytes].copy_(values[src:src + row_bytes])
            else:
                expected[begin:begin + row_bytes].zero_()
        ids = torch.tensor([sources], dtype=torch.int32, device="cuda")
        target = torch.tensor([destinations], dtype=torch.int32, device="cuda")
        attention = torch.full_like(ids, -77)
        if kind == "plan":
            registered_ops.hisparse_gather_plan(host, hot, ids, target,
                torch.ones_like(ids), None, attention, block_size + 64)
        else:
            counts = torch.tensor([3], dtype=torch.int32, device="cuda")
            registered_ops.hisparse_gather_compact(host, hot, ids, target, counts)
        torch.cuda.synchronize()
        torch.testing.assert_close(owner.cpu(), expected, rtol=0, atol=0)
        if kind == "plan":
            indices = [[x // block_size * (block_size + 64) + x % block_size
                        for x in destinations]]
            torch.testing.assert_close(attention.cpu(), torch.tensor(indices,
                                       dtype=torch.int32), rtol=0, atol=0)


def refused_call(ops, host, kind, zero_work=False, cpu_indices=False):
    """All destination and replacement state must be untouched on rejection."""
    hot = torch.full((3, 64, 16), 85, dtype=torch.uint8, device="cuda")
    shape = (0 if zero_work else 1, 1)
    ids = torch.zeros(shape, dtype=torch.int32, device="cuda")
    target = torch.full_like(ids, -77)
    attention = torch.full_like(ids, -77)
    dgi = torch.full((1, 4), -1, dtype=torch.int32, device="cuda")
    lru = torch.arange(4, dtype=torch.int16, device="cuda").view(1, 4)
    tensors = (hot, target, attention, dgi, lru)
    before = [t.clone() for t in tensors]
    if cpu_indices:
        ids = ids.cpu()
    with pytest.raises(RuntimeError, match="host_cache|same one HIP|share one HIP"):
        if kind == "plan":
            ops.hisparse_gather_plan(host, hot, ids, target, torch.ones_like(target),
                                     None, attention, 64)
        elif kind == "compact":
            counts = torch.ones(shape[0], dtype=torch.int32, device="cuda")
            ops.hisparse_gather_compact(host, hot, ids, target, counts)
        else:
            table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
            state = torch.tensor([0], dtype=torch.int32, device="cuda")
            ops.hisparse_resolve_residency(host, hot, table, ids, target,
                                           dgi, lru, state, 4, 4)
    torch.cuda.synchronize()
    for tensor, expected in zip(tensors, before):
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["plan", "compact", "resolve"])
@pytest.mark.parametrize("zero_work", [False, True])
def test_pageable_is_rejected_even_without_work(registered_ops, kind, zero_work):
    refused_call(registered_ops, torch.empty((8, 16), dtype=torch.uint8), kind,
                 zero_work=zero_work)


@pytest.mark.parametrize("kind", ["plan", "compact", "resolve"])
@pytest.mark.parametrize("bad", ["noncontiguous", "past_end", "gap", "released",
                                  "gpu_source", "empty_pageable", "cpu_indices"])
def test_invalid_registered_span_or_device_is_rejected(registered_ops, kind, bad):
    regions = [(0, 1), (2, 1)] if bad == "gap" else [(0, 1)]
    with registration(regions=regions) as (pool, live):
        if bad == "noncontiguous":
            host = pool[:256].view(16, 16)[::2]
        elif bad == "past_end":
            # Start is registered, but the contiguous view extends past its end.
            host = pool[16:16 + 4096].view(-1, 16)
        elif bad == "gap":
            # Both ends are registered. An endpoint-only check is insufficient.
            host = pool[:3 * 4096].view(-1, 16)
        elif bad == "released":
            host = pool[:128].view(8, 16)
            torch.cuda.synchronize()
            assert torch.cuda.cudart().cudaHostUnregister(live.pop()).value == 0
        elif bad == "gpu_source":
            host = torch.empty((8, 16), dtype=torch.uint8, device="cuda")
        elif bad == "empty_pageable":
            host = torch.empty((0, 16), dtype=torch.uint8)
        else:
            host = pool[:128].view(8, 16)
        refused_call(registered_ops, host, kind, cpu_indices=bad == "cpu_indices")
