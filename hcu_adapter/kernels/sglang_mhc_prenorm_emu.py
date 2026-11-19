import torch


def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
    """DeepGEMM split-K contract using full-FP32 torch GEMMs on HCU."""
    if a.ndim != 2 or b.ndim != 2 or d.ndim not in (2, 3) or sqr_sum.ndim not in (1, 2):
        raise ValueError("mHC prenorm GEMM rank mismatch")
    _check((a, torch.bfloat16), (b, torch.float32), (d, torch.float32), (sqr_sum, torch.float32))
    tokens, k = a.shape
    width, k_b = b.shape
    splits = 1 if num_splits is None else num_splits
    if k != k_b or k <= 0 or k % 64 or not 0 < width <= 128 or width % 8 or splits < 1:
        raise ValueError("mHC prenorm GEMM requires K%64=0, 0<N<=128, N%8=0 and positive splits")
    if any(t.stride(-1) != 1 for t in (a, b, d)) or not sqr_sum.is_contiguous():
        raise ValueError("mHC prenorm GEMM requires dense K/N rows and contiguous square sums")
    expected_d = (tokens, width) if num_splits is None else (splits, tokens, width)
    expected_s = (tokens,) if num_splits is None else (splits, tokens)
    _shape(d, expected_d)
    _shape(sqr_sum, expected_s)
    if tokens == 0:
        return
    num_k_blocks = k // 64
    blocks_per_split, k_remain = divmod(num_k_blocks, splits)
    outputs = d.unsqueeze(0) if num_splits is None else d
    squares = sqr_sum.unsqueeze(0) if num_splits is None else sqr_sum
    start = 0
    for first, count, block_count in (
        (0, k_remain, blocks_per_split + 1),
        (k_remain, splits - k_remain, blocks_per_split),
    ):
        if count == 0:
            continue
        output = outputs[first:first + count]
        square = squares[first:first + count]
        if block_count:
            split_k = block_count * 64
            end = start + count * split_k
            lhs = a[:, start:end].float().reshape(tokens, count, split_k).permute(1, 0, 2)
            rhs = b[:, start:end].reshape(width, count, split_k).permute(1, 2, 0)
            output.copy_(torch.bmm(lhs, rhs))
            square.copy_(lhs.square().sum(-1))
            start = end
        else:
            output.zero_()
            square.zero_()

def _check(*typed_tensors):
    device = typed_tensors[0][0].device
    for tensor, dtype in typed_tensors:
        if tensor.device != device or device.type != "cuda":
            raise ValueError("mHC tensors must share a CUDA/HIP device")
        if tensor.dtype != dtype:
            raise TypeError("mHC tensor dtype does not match the pinned inference contract")

def _shape(tensor, expected):
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError("mHC tensor shape mismatch: " + str(tuple(tensor.shape)) + " != " + str(tuple(expected)))
