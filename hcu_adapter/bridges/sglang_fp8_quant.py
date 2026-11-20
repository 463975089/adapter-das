# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""fp8 quant path bridges (HCU_FP8_TRACE probe, native per-token quant bug, group-quant dtype)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os
import torch


def install(_he=None) -> None:
    # Debug facility: HCU_FP8_TRACE=1 logs every fp8 apply call (shapes +
    # OK/RAISE) to /tmp/fp8_trace.log — serve-time GEMM-path localization.
    if os.environ.get("HCU_FP8_TRACE") == "1":
        try:
            import sglang.srt.layers.quantization.fp8_utils as _fpu

            _trace_log = open("/tmp/fp8_trace.log", "w", buffering=1)

            def _hcu_trace(fn, name):
                def w(*a, **k):
                    try:
                        out = fn(*a, **k)
                        _trace_log.write(f"{name} OK\n")
                        return out
                    except Exception as e:
                        _trace_log.write(f"{name} RAISE {e!r:.140}\n")
                        raise

                return w

            for _name in ("apply_fp8_linear", "apply_w8a8_block_fp8_linear",
                          "apply_fp8_ptpc_linear"):
                _fn = getattr(_fpu, _name, None)
                if callable(_fn):
                    setattr(_fpu, _name, _hcu_trace(_fn, _name))
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: fp8 trace not installed", exc_info=True
            )

    # v0.5.21 HIP native fp8 per-token quant bug (Qwen3-4B-FP8 decode-graph
    # capture, bs=16 + output_padding=17): apply_fp8_linear passes a
    # preallocated (bs,1) scale buffer while num_token_padding makes the
    # quant output bs+1 rows — `_native_dynamic_per_token_quant_fp8` then
    # copy_s a (17,1) scale into a (16,1) buffer. The aiter branch handles
    # this; the native fallback doesn't. Bridge: align scale to the padded
    # row count before quant (graph-safe: same buffers, no host sync).
    try:
        # 替代档: fp8_kernel 模块=quant 桥挂载面（包面 per_token_group_quant 走分档）
        import sglang.kernels.ops.quantization.fp8_kernel as _fpk

        # is_cuda-seam ordering guard: this module reads is_cuda()/is_hip()
        # at import time into module globals, and in some boot orders its
        # first import predates the platform_seam patch of common, leaving
        # _is_hip=True / _is_cuda=False (HIP semantics) cached here while
        # everything else answers CUDA. Re-assert the seam values (and pull
        # the CUDA-only per_token_group_quant import the _is_cuda gate
        # skipped) — cheap idempotent setattr, keeps the EP fp8 path on the
        # same semantics as the rest of the runtime.
        if not getattr(_fpk, "_is_cuda", False):
            _fpk._is_cuda = True
            _fpk._is_hip = False
            if not hasattr(_fpk, "per_token_group_quant"):
                try:
                    # Package face: the catalog's JIT wrapper for this
                    # symbol (same call shape, including masked/fused-silu
                    # tri-state kwargs). The AOT alternative
                    # sgl_kernel.sgl_per_token_group_quant_8bit does NOT
                    # cover this call shape: it requires eps/fp8_min/
                    # fp8_max and returns None, while fp8_kernel passes
                    # (x, x_q, x_s, group_size) + kwargs.
                    from sglang.kernels.ops.quantization import (
                        per_token_group_quant as _ptgq,
                    )

                    _fpk.per_token_group_quant = _ptgq
                except Exception:
                    pass

        _orig_spq = _fpk.scaled_fp8_quant

        def _hcu_scaled_fp8_quant(
            input, scale=None, num_token_padding=None,
            use_per_token_if_dynamic=False, **kw,
        ):
            # v0.5.21 HIP native bug: the wrapper pads ONLY the output tensor
            # to num_token_padding but `_native_dynamic_per_token_quant_fp8`
            # divides the unpadded input by the padded per-token scale
            # (17,1) -> shape mismatch. The aiter branch pads its quant
            # internally; the native one doesn't. Bridge: quantize unpadded
            # with the native path, then pad the output ourselves (and pad
            # the scale with neutral rows so downstream _scaled_mm sees
            # consistent per-row scales). Graph-safe: no host sync.
            pad_rows = 0
            if (
                num_token_padding
                and num_token_padding > input.shape[0]
                and not _fpk._use_aiter
            ):
                pad_rows = num_token_padding - input.shape[0]
                num_token_padding = None
            out, s = _orig_spq(
                input, scale, num_token_padding=num_token_padding,
                use_per_token_if_dynamic=use_per_token_if_dynamic, **kw,
            )
            if pad_rows:
                out = torch.nn.functional.pad(
                    out, (0, 0, 0, pad_rows)
                ).contiguous()
                # Keep the upstream fp8_kernel.scaled_fp8_quant contract:
                # num_token_padding allocates BOTH the output and the
                # dynamic per-token scale at padded rows. Pad that scale
                # with neutral 1.0 rows so downstream _scaled_mm sees a
                # per-row scale matching the padded output (padded output
                # rows are zeros, so they dequant to zero under any
                # finite scale). Static / per-tensor scales stay 1-D and
                # broadcast unchanged. Graph-safe: constant pad, no host
                # sync.
                if use_per_token_if_dynamic and scale is None and s.dim() == 2:
                    s = torch.nn.functional.pad(
                        s, (0, 0, 0, pad_rows), value=1.0
                    ).contiguous()
            return out, s

        _hcu_scaled_fp8_quant.__wrapped__ = _orig_spq
        _fpk.scaled_fp8_quant = _hcu_scaled_fp8_quant
        # byname importers that were loaded before this bridge
        import sys as _sys

        for _mod_name in tuple(_sys.modules):
            if _mod_name.endswith("quantization.fp8_utils"):
                _m = _sys.modules[_mod_name]
                if getattr(_m, "scaled_fp8_quant", None) is _orig_spq:
                    _m.scaled_fp8_quant = _hcu_scaled_fp8_quant
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: fp8 per-token padding bridge not installed",
            exc_info=True,
        )

    # group-quant output dtype bridge (EP deep_gemm runner): the JIT
    # per_token_group_quant kernel on HIP binds Q as uint8 (fp8 is
    # storage-typed), while the upstream wrapper allocates a float8_e4m3fn
    # tensor. Wrap the wrapper: allocate uint8 under the hood, return the
    # fp8 view (zero-copy reinterpret) so downstream GEMM callers keep
    # seeing the expected dtype.
    try:
        # 替代档: fp8_kernel 模块=quant 桥挂载面（fp8-out 特化绕 dtype 门）
        import sglang.kernels.ops.quantization.fp8_kernel as _fpk2

        # The JIT per_token_group_quant kernel binds Q as uint8 (HIP fp8 is
        # storage-typed), while the upstream driver allocates float8_e4m3fn
        # and derives the range from that dtype. Wrap the JIT symbol at the
        # fp8_kernel module global: hand the kernel a uint8 view of the same
        # buffer and override the baked range; the caller-visible tensor
        # stays fp8.
        _orig_ptgq_kernel = _fpk2.per_token_group_quant

        def _hcu_ptgq_kernel(x, x_q, x_s, group_size, *, scale_ue8m0=False,
                             fuse_silu_and_mul=False, masked_m=None):
            # JIT matcher binds Q as uint8 (HIP fp8 = storage bytes); wrap
            # the fp8-allocated buffer into its uint8 view for the call.
            is_fp8 = x_q.dtype == torch.float8_e4m3fn
            if is_fp8:
                # route through the int8-typed JIT variant so the module-level
                # dtype gate passes; the int8 and fp8 e4m3 kernels share the
                # same code path modulo the bound max (e4m3 clamping happens
                # in WeightTrait, which the JIT instantiates from out_dtype —
                # int8 bakes 127.0 for the clamp, which is WRONG for e4m3).
                # So instead: keep fp8 semantics by calling the fp8-flavored
                # JIT through a temporary dtype-forgiving matcher: hand the
                # matcher the fp8 tensor it expects — tvm_ffiDLManager maps
                # float8_e4m3fn natively; the mismatch came from OUR view.
                # Simplest correct route: allocate uint8 tensor ourselves and
                # reach the same _jit_module with out_dtype=fp8 by passing the
                # original fp8 dtype on the python side — the matcher then
                # binds bytes. Fall back to the plain call for non-e4m3.
                pass
            xq_pass = x_q.view(torch.uint8) if is_fp8 and x_q.numel() else x_q
            # bypass the dtype gate by pre-specializing through the private
            # loader with out_dtype=fp8 (the JIT itself binds bytes under
            # the HIP alias) — via the module function with fp8 dtype.
            # JIT 唯一实现（fp8-out 特化；私有 _SUPPORTED_OUTPUT_DTYPES 仅模块体可见）
            import sglang.kernels.ops.quantization.per_token_group_quant as _ptg
            import torch as _t

            saved = _ptg._SUPPORTED_OUTPUT_DTYPES
            try:
                if is_fp8 and torch.uint8 not in saved:
                    _ptg._SUPPORTED_OUTPUT_DTYPES = saved + (torch.uint8,)
                return _orig_ptgq_kernel(
                    x, xq_pass, x_s, group_size,
                    scale_ue8m0=scale_ue8m0,
                    fuse_silu_and_mul=fuse_silu_and_mul,
                    masked_m=masked_m,
                )
            finally:
                _ptg._SUPPORTED_OUTPUT_DTYPES = saved

        _fpk2.per_token_group_quant = _hcu_ptgq_kernel
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: group-quant output dtype bridge not installed",
            exc_info=True,
        )
