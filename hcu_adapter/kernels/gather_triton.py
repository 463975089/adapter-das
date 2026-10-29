"""Triton implementation of the existing gather contract."""

import triton
import triton.language as tl


@triton.jit
def _gather(X, Index, Y, N: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
            XS: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offset // D
    column = offset % D
    index = tl.load(Index + row, row < M, other=-1).to(tl.int64)
    value = tl.load(X + index * XS + column,
                    (row < M) & (index >= 0) & (index < N), other=0)
    tl.store(Y + offset, value, offset < M * D)


def gather(x, index, out):
    import torch

    with torch.cuda.device(x.device):
        _gather[(triton.cdiv(out.numel(), 1024),)](
            x, index, out, x.shape[0], x.shape[1], index.numel(), x.stride(0), 1024)
