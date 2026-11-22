# SPDX-License-Identifier: Apache-2.0
"""Native BF16 inverse RoPE together with empty-row writeback."""
from __future__ import annotations

import logging
import os

import torch
from hcu_adapter.operator_contracts import supports_operator_contract

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_LOG = logging.getLogger('vllm.hcu_adapter')
_qualified_inverse_rope = set()


def _native_entries(native):
    return (getattr(native, '_inverse_rope_gptj_kernel', None),
            getattr(native, 'rocm_inverse_rope_rows_', None))


def _native_matches(device):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native

    entries = _native_entries(native)
    return all(callable(entry) for entry in entries) and (
        (os.getpid(), device, *entries) in _qualified_inverse_rope
        or supports_operator_contract(native, 'mla_bf16_inverse_rope'))


if triton is not None:
    @triton.jit
    def _mask_copy_inverse_rope(Result, Output, Main, Extra, Positions, CS,
                                HEADS: tl.constexpr, HAS_EXTRA: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        base = (row * HEADS + head) * 512
        count = tl.load(Main + row)
        if HAS_EXTRA:
            count += tl.load(Extra + row)
        n = tl.arange(0, 512)
        value = tl.load(Result + base + n, (n < 448) & (count > 0), other=0.)
        tl.store(Output + base + n, value, n < 448)
        pos = tl.load(Positions + row).to(tl.int64)
        k = tl.arange(0, 32)
        a = tl.load(Result + base + 448 + 2 * k, count > 0, other=0.).to(tl.float32)
        b = tl.load(Result + base + 448 + 2 * k + 1, count > 0, other=0.).to(tl.float32)
        cos = tl.load(CS + pos * 64 + k)
        sin = tl.load(CS + pos * 64 + 32 + k)
        # Keep the native expression and default fused arithmetic.
        # Empty rows rotate positive BF16 zeros too, preserving signed zeros.
        even = a * cos + b * sin
        odd = b * cos - a * sin
        tl.store(Output + base + 448 + 2 * k, even.to(tl.bfloat16))
        tl.store(Output + base + 448 + 2 * k + 1, odd.to(tl.bfloat16))


@torch.inference_mode()
def qualify_inverse_rope(device):
    """Check the installed inverse RoPE against fused BF16 writeback before capture."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native

    device = torch.device(device)
    if triton is None or device.type != 'cuda':
        return False
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '').split(':')[0] != 'gfx938':
        return False
    entries = _native_entries(native)
    if not all(callable(entry) for entry in entries):
        return False
    if _native_matches(device):
        return True
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            angle = torch.arange(256 * 32, device=device, dtype=torch.float32).reshape(256, 32) / 61
            cos_sin = torch.cat((angle.cos(), angle.sin()), dim=1)
            for rows, heads in ((1, 64), (8, 128), (48, 64), (64, 128), (256, 64), (256, 128)):
                # Include every BF16 encoding at the largest shape.
                bits = torch.arange(rows * heads * 512, device=device, dtype=torch.int32)
                result = bits.to(torch.int16).view(torch.bfloat16).reshape(rows, heads, 512)
                output = torch.empty_like(result)
                row = torch.arange(rows, device=device, dtype=torch.int32)
                for with_extra in (False, True):
                    main = row % 3 - 1
                    extra = row % 2 if with_extra else None
                    live = main > 0 if extra is None else main + extra > 0
                    positions = (255 - row).to(torch.int64) if with_extra else 255 - row
                    expected = torch.where(live[:, None, None], result, 0.)
                    entries[1](expected, positions, cos_sin, 64)
                    _mask_copy_inverse_rope[(rows, heads)](
                        result, output, main, extra, positions, cos_sin,
                        heads, with_extra, num_warps=4)
                    if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
                        _LOG.warning('HCU inverse RoPE preflight output differs from vLLM Core')
                        return False
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as error:
            _LOG.warning('HCU inverse RoPE preflight unavailable: %s', error)
            return False
    _qualified_inverse_rope.add((os.getpid(), device, *entries))
    _LOG.info('HCU fused MLA inverse RoPE writeback admitted after native preflight')
    return True


def try_mask_copy_rope(result, output, main_lens, extra_lens, positions, cos_sin):
    if triton is None or positions is None or cos_sin is None:
        return False
    tensors = (result, output, main_lens, positions, cos_sin) + (() if extra_lens is None else (extra_lens,))
    if (result.ndim != 3 or not 0 < result.shape[0] <= 256
            or result.shape[1:] not in ((64, 512), (128, 512))
            or output is None or output.shape != result.shape
            or result.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or not all(t.is_cuda and t.device == result.device and t.is_contiguous() for t in tensors)
            or main_lens.shape != (result.shape[0],) or main_lens.dtype != torch.int32
            or (extra_lens is not None and (extra_lens.shape != main_lens.shape or extra_lens.dtype != torch.int32))
            or positions.shape != main_lens.shape or positions.dtype not in (torch.int32, torch.int64)
            or cos_sin.ndim != 2 or cos_sin.shape[0] == 0
            or cos_sin.shape[1] != 64 or cos_sin.dtype != torch.float32):
        return False
    if getattr(torch.cuda.get_device_properties(result.device), 'gcnArchName', '').split(':')[0] != 'gfx938':
        return False
    if not _native_matches(result.device):
        return False
    _mask_copy_inverse_rope[(result.shape[0], result.shape[1])](
        result, output, main_lens, extra_lens, positions, cos_sin,
        result.shape[1], extra_lens is not None, num_warps=4)
    return True
