"""Bound the injected Torch post's row-local temporary memory."""
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime


def reference(x, residual, post, comb):
    mixed = torch.einsum('...ij,...ih->...jh', comb.float(), residual.float())
    return (mixed + post.float() * x.unsqueeze(-2).float()).to(residual.dtype)


@pytest.mark.parametrize('tokens', [0, 1, 1024, 1025, 2051])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('strided', [False, True])
def test_chunked_post_preserves_rows_dtype_and_native_arithmetic(tokens, dtype, strided):
    torch.manual_seed(17)
    stride = 2 if strided else 1
    x = torch.randn(tokens, 17 * stride, dtype=dtype)[:, ::stride]
    residual = torch.randn(tokens, 4, 17 * stride, dtype=dtype)[..., ::stride]
    post = torch.randn(tokens, 4, 1)
    comb = torch.randn(tokens, 4, 4)
    calls = []

    def injected(*args):
        calls.append(args[0].shape[0])
        return reference(*args)

    runtime = MhcAiterRuntime(MhcAiterProvider(),
        SimpleNamespace(mhc_post_torch=injected), None, None, None)
    actual = runtime.post(x, residual, post, comb)
    torch.testing.assert_close(actual, reference(x, residual, post, comb), rtol=0, atol=0)
    assert actual.dtype == dtype
    assert sum(calls) == tokens
    assert max(calls, default=0) <= 1024


def test_broadcast_mix_keeps_the_native_entry():
    x = torch.randn(1025, 17)
    residual = torch.randn(1025, 4, 17)
    post = torch.randn(1025, 4, 1)
    comb = torch.randn(4, 4)
    calls = []

    def injected(*args):
        calls.append(args[0].shape[0])
        return reference(*args)

    runtime = MhcAiterRuntime(MhcAiterProvider(),
        SimpleNamespace(mhc_post_torch=injected), None, None, None)
    torch.testing.assert_close(runtime.post(x, residual, post, comb),
                               reference(x, residual, post, comb), rtol=0, atol=0)
    assert calls == [1025]
