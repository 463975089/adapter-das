# SPDX-License-Identifier: Apache-2.0
"""Real HIP byte-copy coverage for BF16 and opaque FP8-DS MLA cache rows.

These are transfer/layout checks, not numerical attention/quantization tests.
"""
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(torch.version.hip is None, reason="requires HIP Torch")


@pytest.fixture(scope="module")
def typed_native(request):
    if not torch.cuda.is_available():
        pytest.skip("requires an available HIP device; NOT qualified")
    from hcu_adapter.kernels.hisparse import ensure_hisparse_ops
    ensure_hisparse_ops(request.config.getoption("--hisparse-library"))
    with torch.cuda.device(request.config.getoption("--hisparse-device")):
        yield torch.ops._C_cache_ops


@pytest.mark.parametrize("dtype,columns", [(torch.bfloat16, 576), (torch.uint8, 656)])
@pytest.mark.parametrize("block_size", [64, 256])
@pytest.mark.parametrize("kind", ["plan", "compact"])
def test_typed_cache_copy_preserves_payload_and_guards(typed_native, dtype, columns, block_size, kind):
    element_size = torch.empty((), dtype=dtype).element_size()
    row_bytes = columns * element_size
    rows, blocks, offset, gap, guard = 8, 3, 2, 24, 32
    host_owner = torch.empty(offset + rows * row_bytes, dtype=torch.uint8, pin_memory=True)
    host_owner.copy_((torch.arange(host_owner.numel(), dtype=torch.int64) * 17 % 251).to(torch.uint8))
    host = host_owner[offset:].view(dtype).view(rows, columns)
    assert host.is_pinned() and host.storage_offset() > 0
    pitch = block_size * row_bytes + gap
    size = offset + blocks * pitch + guard
    owner = torch.full((size,), 85, dtype=torch.uint8, device="cuda")
    hot = owner.view(dtype).as_strided(
        (blocks, block_size, columns), (pitch // element_size, columns, 1),
        offset // element_size,
    )
    expected = torch.full((size,), 85, dtype=torch.uint8)
    globals_ = [0, 3, 999]
    destinations = [block_size - 1, block_size, blocks * block_size - 1]
    assert len(set(destinations)) == len(destinations)
    for source, target in zip(globals_, destinations):
        start = offset + (target // block_size) * pitch + (target % block_size) * row_bytes
        if source < rows:
            begin = offset + source * row_bytes
            expected[start:start + row_bytes].copy_(host_owner[begin:begin + row_bytes])
        else:
            expected[start:start + row_bytes].zero_()
    ids = torch.tensor([globals_], dtype=torch.int32, device="cuda")
    targets = torch.tensor([destinations], dtype=torch.int32, device="cuda")
    if kind == "compact":
        counts = torch.tensor([3], dtype=torch.int32, device="cuda")
        typed_native.hisparse_gather_compact(host, hot, ids, targets, counts)
    else:
        masks = torch.ones_like(ids)
        attention = torch.empty_like(ids)
        stride = block_size + 64
        typed_native.hisparse_gather_plan(host, hot, ids, targets, masks, None, attention, stride)
    # Retain pinned source and all views until the producing stream is finished.
    torch.cuda.synchronize()
    torch.testing.assert_close(owner.cpu(), expected, rtol=0, atol=0)
    if kind == "plan":
        expected_indices = torch.tensor([[x // block_size * stride + x % block_size
                                          for x in destinations]], dtype=torch.int32)
        torch.testing.assert_close(attention.cpu(), expected_indices, rtol=0, atol=0)
