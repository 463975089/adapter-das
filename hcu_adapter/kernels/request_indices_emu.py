"""Expand live request rows and map padded tokens to reserved request zero."""

import torch


def padded_request_indices(requests, lengths, output_size):
    """Device-only equivalent of repeat_interleave plus zero padding.

    SGLang reserves request row zero for padding. Unlike repeat_interleave's
    output_size, this size may exceed the sum of the live extend lengths.
    """
    if requests.ndim != 1 or lengths.ndim != 1 or requests.shape != lengths.shape:
        raise ValueError("request indices and lengths must be matching vectors")
    if requests.dtype != torch.int64 or lengths.dtype != torch.int64:
        raise TypeError("request indices and lengths must be int64")
    if requests.device != lengths.device or not isinstance(output_size, int) or output_size < 0:
        raise ValueError("request vectors need the same device and a nonnegative size")
    ends = lengths.cumsum(0)
    valid = torch.all(lengths >= 0) & (lengths.sum() <= output_size)
    message = "extend lengths must be nonnegative and fit the padded token count"
    if requests.is_cuda:
        torch._assert_async(valid, message)
    elif not bool(valid):
        raise ValueError(message)
    rows = torch.arange(output_size, device=requests.device, dtype=torch.int64)
    owners = torch.searchsorted(ends, rows, right=True)
    # The sentinel also covers empty batches and zero-length trailing requests.
    request_with_padding = torch.cat((requests, requests.new_zeros(1)))
    return request_with_padding[owners]
