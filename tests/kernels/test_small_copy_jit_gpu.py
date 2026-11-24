import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _operator():
    from hcu_adapter.features.jit.small_copy_jit import install
    install()
    from sglang.kernels.ops.memory.small_copy import try_small_copy, _small_copy_kernel
    if (type(_small_copy_kernel).__module__, type(_small_copy_kernel).__name__) == (
        "triton.backends.hcu.jit", "FastJITFunction"
    ):
        from triton.runtime.jit import JITFunction
        standard = getattr(_small_copy_kernel, "_hcu_small_copy_standard_jit", None)
        assert isinstance(standard, JITFunction)
        assert _small_copy_kernel.run.__self__ is standard
    return try_small_copy


def test_dynamic_batch_and_pointer_copies():
    op = _operator()
    for n in [1, 2, 2, 16, 1, 2, 0]:
        lengths = [4 * n, 4 * n, 4 * n, n, n]
        sources = [torch.arange(k, device="cuda", dtype=torch.int32 if i == 4 else torch.int64) + 100 + i for i, k in enumerate(lengths)]
        dests = [torch.full((262144,), -999, device="cuda", dtype=torch.int64)[:k] for k in lengths]
        assert op(dests, sources)
        for src, dst in zip(sources, dests):
            torch.testing.assert_close(dst, src.to(dst.dtype), rtol=0, atol=0)


def test_strides_cast_and_changed_input_graph():
    op = _operator()
    sources = [
        torch.arange(48, device="cuda", dtype=torch.int64).reshape(4, 12)[:, ::2],
        torch.arange(9, device="cuda", dtype=torch.int32),
        torch.arange(9, device="cuda", dtype=torch.int64),
    ]
    bases = [torch.full((4, 18), -777, device="cuda", dtype=torch.int64),
             torch.empty(9, device="cuda", dtype=torch.int64),
             torch.empty(9, device="cuda", dtype=torch.int32)]
    dests = [bases[0][:, ::3], bases[1], bases[2]]
    assert op(dests, sources)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert op(dests, sources)
    for change in [3, 17, -9]:
        for src in sources:
            src.add_(change)
        graph.replay()
        for src, dst in zip(sources, dests):
            torch.testing.assert_close(dst, src.to(dst.dtype), rtol=0, atol=0)
        torch.testing.assert_close(bases[0][:, 1::3], torch.full_like(bases[0][:, 1::3], -777), rtol=0, atol=0)
        torch.testing.assert_close(bases[0][:, 2::3], torch.full_like(bases[0][:, 2::3], -777), rtol=0, atol=0)
