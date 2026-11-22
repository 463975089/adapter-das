# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned sparse selector preparation; preserve the eager sort order.

Native sparse decode imports this module for opted-in, bounded CUDA inputs.
One fused program per row handles preparation, sorting and padded output.
Outside the fusion envelope, the three-step path uses a preparation kernel,
``torch.sort`` and a padded-output kernel. ``_FastLaunch`` optionally replays
compiled launches after the first dispatch.
"""
from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl
from triton.runtime import driver

LOG = logging.getLogger("hcu.adapter.kernels.mla_selectors")

# Triton specialises tensor arguments on their 16-byte pointer alignment and
# integer arguments on their divisibility, so both are re-checked per launch.
_ALIGN = 16
_FAST_LAUNCH_ENV = "HCU_TRITON_FAST_LAUNCH"
# Bound the per-row bitonic sort and total work for fusion.
_FUSED_MAX_BLOCK = 4096
_FUSED_MAX_ELEMS = 262144


@triton.jit
def _prepare_selectors(
    values, indptr, lengths, selected, counts,
    VALUE_ROW_STRIDE: tl.constexpr, VALUE_COL_STRIDE: tl.constexpr,
    INDPTR_STRIDE: tl.constexpr, LENGTH_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr, CAPACITY: tl.constexpr, NUM_SLOTS: tl.constexpr,
    IS_CSR: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    inside = col < WIDTH
    if IS_CSR:
        start = tl.load(indptr + row * INDPTR_STRIDE)
        end = tl.load(indptr + (row + 1) * INDPTR_STRIDE)
        # The eager expression subtracts int32 indptr entries before the
        # comparison, but promotes start to int64 before adding columns.
        length = (end - start).to(tl.int32)
        offset = start.to(tl.int64) + col
        in_storage = (offset >= 0) & (offset < CAPACITY)
        value = tl.load(values + offset * VALUE_COL_STRIDE,
                        mask=inside & in_storage, other=-1)
        live = inside & (col < length) & in_storage
    else:
        length = tl.load(lengths + row * LENGTH_STRIDE)
        value = tl.load(values + row * VALUE_ROW_STRIDE + col * VALUE_COL_STRIDE,
                        mask=inside, other=-1)
        live = inside & (col < length)
    valid = live & (value >= 0) & (value.to(tl.int64) < NUM_SLOTS)
    tl.store(selected + row * WIDTH + col,
             tl.where(valid, value, 2147483647), mask=inside)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _write_padded(compact, padded, WIDTH: tl.constexpr,
                  PADDED_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    value = tl.load(compact + row * WIDTH + col, mask=col < WIDTH,
                    other=2147483647)
    tl.store(padded + row * PADDED_WIDTH + col,
             tl.where(value == 2147483647, -1, value), mask=col < PADDED_WIDTH)


@triton.jit
def _empty_selectors(padded, counts, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    tl.store(padded + row * 64 + col, -1, mask=col < 64)
    tl.store(counts + row, 0)


_SENTINEL = tl.constexpr(2147483647)


@triton.jit
def _selectors_fused(
    values, indptr, lengths, padded, counts,
    VALUE_ROW_STRIDE: tl.constexpr, VALUE_COL_STRIDE: tl.constexpr,
    INDPTR_STRIDE: tl.constexpr, LENGTH_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr, CAPACITY: tl.constexpr, NUM_SLOTS: tl.constexpr,
    IS_CSR: tl.constexpr, BLOCK: tl.constexpr, PADDED_WIDTH: tl.constexpr,
):
    """Prepare, sort ascending and pad one row in a single program.

    Same result as ``_prepare_selectors`` followed by ``selected.sort(dim=-1)``
    and ``_write_padded``: sorting equal ids is order-free and every sentinel
    becomes -1, so one ascending bitonic sort reproduces the torch result bit
    for bit while dropping two launches and the ``(rows, width)`` intermediate.
    """
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    inside = col < WIDTH
    if IS_CSR:
        start = tl.load(indptr + row * INDPTR_STRIDE)
        end = tl.load(indptr + (row + 1) * INDPTR_STRIDE)
        length = (end - start).to(tl.int32)
        offset = start.to(tl.int64) + col
        in_storage = (offset >= 0) & (offset < CAPACITY)
        value = tl.load(values + offset * VALUE_COL_STRIDE,
                        mask=inside & in_storage, other=-1)
        live = inside & (col < length) & in_storage
    else:
        length = tl.load(lengths + row * LENGTH_STRIDE)
        value = tl.load(values + row * VALUE_ROW_STRIDE + col * VALUE_COL_STRIDE,
                        mask=inside, other=-1)
        live = inside & (col < length)
    valid = live & (value >= 0) & (value.to(tl.int64) < NUM_SLOTS)
    ids = tl.sort(tl.where(valid, value, _SENTINEL))
    tl.store(padded + row * PADDED_WIDTH + col,
             tl.where(ids == _SENTINEL, -1, ids), mask=col < PADDED_WIDTH)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


def _fast_launch_enabled():
    """Registry default off. Direct file probes honour the raw env."""
    try:
        from hcu_adapter.infra.hcu_env import enabled
    except ImportError:  # pragma: no cover - probes import this file directly
        return os.environ.get(_FAST_LAUNCH_ENV, "0") not in ("", "0", "None", "refused")
    return enabled(_FAST_LAUNCH_ENV)


def _arg_key(args, ntensors):
    """Cheap per-launch key of everything the cached specialisation depends on.

    ``torch.is_tensor`` plus a generator costs ~15 ``is_tensor`` and ~34
    ``isinstance`` buckets per selectors call in the stacked 20261118 trace, so
    the call sites pass how many leading arguments are tensors (they always
    are, in this module) instead of probing every argument's type here.
    """
    key = [(a.dtype, a.data_ptr() & (_ALIGN - 1)) for a in args[:ntensors]]
    key.extend(args[ntensors:])
    return tuple(key)


def _replay(kernel, launcher, grid, args):
    """The launch tail of ``JITFunction.run`` with the dispatch already done.

    The compiled kernel, its launcher handle, the packed metadata and the
    canonical grid are the inputs ``JITFunction.run`` resolves on every call.
    The stream, the device and the profiler launch hooks stay live here, so
    trace correlation and capture semantics are exactly the upstream ones.
    """
    from triton import knobs

    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    launcher(
        grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1,
        stream, kernel.function, kernel.packed_metadata,
        kernel.launch_metadata(grid, stream, *args),
        knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
        *args,
    )


class _FastLaunch:
    """Replay a captured Triton launch, bypassing the generic JIT dispatch.

    ``JITFunction.run`` rebinds the full signature (11.9 us/call measured on
    triton 3.7.0/HCU in this image), rebuilds the specialisation and redoes
    the compiled-kernel cache lookup before every launch. These kernels have
    one call site each with a fixed argument layout, so the first launch can
    hand over its compiled kernel and launcher.

    Nothing observable changes: the arguments passed to the kernel are the
    current call's objects, the grid is the current call's grid, the current
    stream and profiler hooks are read per call, and the kernels read no module
    globals (so the ``used_global_vals`` guard defends nothing here). Everything
    the cached specialisation depends on - device, tensor dtype, 16-byte pointer
    alignment and integer argument values - keys the cache, so a mismatch is a
    fresh site (captured through the untouched JIT path) rather than a replay.
    Registration hooks installed on the JIT function (``pre_run_hooks``) also
    force the untouched path.
    """

    __slots__ = ("_fn", "_sites", "_off", "fallbacks", "_warned")

    # The batch size is a launch grid, not an argument: ``_write_padded`` keeps
    # one site for every batch size, while ``_prepare_selectors`` gets one per
    # CSR capacity (rows * width is a constexpr) - i.e. potentially one per
    # distinct decode batch. Sites are a dict entry each, so the bound only
    # guards against an unbounded shape stream and is deliberately generous.
    _MAX_SITES = 512

    def __init__(self, jit_fn):
        self._fn = jit_fn
        self._sites = {}
        self._off = None
        self.fallbacks = 0
        self._warned = False

    def __call__(self, grid, args, ntensors):
        if self._off is None:
            self._off = not _fast_launch_enabled()
        if self._off or self._fn.pre_run_hooks:
            return self._fn[grid](*args)
        key = _arg_key(args, ntensors)
        site = self._sites.get(key)
        if site is None:
            if len(self._sites) >= self._MAX_SITES:
                self._latch(
                    "site table full (%d) on %s; using the generic JIT path",
                    len(self._sites), getattr(self._fn, "__name__", self._fn),
                )
                return self._fn[grid](*args)
            return self._capture(grid, args, key)
        kernel, launcher, device = site
        if driver.active.get_current_device() != device:
            self._latch(
                "current device changed on %s; using the generic JIT path",
                getattr(self._fn, "__name__", self._fn),
            )
            return self._fn[grid](*args)
        try:
            _replay(kernel, launcher, grid, args)
        except Exception:  # pragma: no cover - triton ABI drift
            self._sites.pop(key, None)
            self._latch(
                "replay failed on %s; re-arming through the JIT path",
                getattr(self._fn, "__name__", self._fn),
            )
            return self._fn[grid](*args)
        return kernel

    def _latch(self, msg, *fmt):
        """Count every fallback, but log the first one only."""
        self.fallbacks += 1
        if not self._warned:
            self._warned = True
            LOG.warning("triton fast launch fallback: " + msg, *fmt)

    def _capture(self, grid, args, key):
        kernel = self._fn[grid](*args)
        self._sites[key] = (kernel, kernel.run,
                            driver.active.get_current_device())
        LOG.info(
            "triton fast launch armed for %s: grid=%s sites=%d",
            getattr(self._fn, "__name__", self._fn), tuple(grid),
            len(self._sites),
        )
        return kernel


_PREPARE_LAUNCH = _FastLaunch(_prepare_selectors)
_WRITE_LAUNCH = _FastLaunch(_write_padded)
_FUSED_LAUNCH = _FastLaunch(_selectors_fused)


def selectors(dense, lengths, flat, indptr, rows, num_slots, *, fused=True):
    """Return exactly the eager padded int32 ids and counts for bounded rows.

    All input strides are honored. CSR storage capacity, not indptr[-1],
    determines the host-known row width. There are no device-to-host reads.
    """
    is_csr = flat is not None
    if is_csr:
        values = flat
        width = flat.numel() // rows
        row_stride, col_stride = 0, flat.stride(0)
        indptr_stride, length_stride = indptr.stride(0), 0
        # Unused pointer on the constexpr CSR branch.
        lengths_arg = flat
    else:
        # Retain the eager reshape behavior for 3D/non-contiguous dense views.
        values = dense.reshape(rows, -1)
        width = values.shape[1]
        row_stride, col_stride = values.stride()
        indptr_stride, length_stride = 0, lengths.stride(0)
        indptr = values
        lengths_arg = lengths
    padded_width = max(64, (width + 63) // 64 * 64)
    # tl.sort needs a power-of-two block that also covers the padded row.
    block = max(triton.next_power_of_2(width),
                triton.next_power_of_2(padded_width))
    with torch.cuda.device(values.device):
        padded = torch.empty((rows, 1, padded_width), dtype=torch.int32,
                             device=values.device)
        counts = torch.empty((rows,), dtype=torch.int32, device=values.device)
        if width == 0:
            _empty_selectors[(rows,)](padded, counts, BLOCK=64)
            return padded, counts
        capacity = flat.numel() if is_csr else 0
        if fused and block <= _FUSED_MAX_BLOCK and rows * block <= _FUSED_MAX_ELEMS:
            _FUSED_LAUNCH((rows,), (
                values, indptr, lengths_arg, padded, counts,
                row_stride, col_stride, indptr_stride, length_stride,
                width, capacity, num_slots, is_csr, block, padded_width,
            ), 5)
            return padded, counts
        selected = torch.empty((rows, width), dtype=torch.int32, device=values.device)
        _PREPARE_LAUNCH((rows,), (
            values, indptr, lengths_arg, selected, counts,
            row_stride, col_stride, indptr_stride, length_stride,
            width, capacity, num_slots, is_csr,
            triton.next_power_of_2(width),
        ), 5)
        compact = selected.sort(dim=-1).values
        _WRITE_LAUNCH((rows,), (
            compact, padded, width, padded_width,
            triton.next_power_of_2(padded_width),
        ), 2)
    return padded, counts
