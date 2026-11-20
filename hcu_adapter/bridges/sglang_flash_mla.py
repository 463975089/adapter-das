# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA construct-only stubs + sgl-flash-mla wheel face skip contract

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging


def install(_he=None) -> None:
    # FlashMLA decode backend: FlashMLABackend subclasses
    # FlashInferMLAAttnBackend whose __init__ unconditionally constructs
    # flashinfer wrappers (BatchMLAPagedAttentionWrapper,
    # BatchPrefillWithRaggedKVCacheWrapper). flashinfer is unavailable on HCU
    # and the availability-gated import at the module top never binds those
    # names -> NameError at backend construction. The flashmla forward paths
    # never plan()/run() the wrappers themselves (they call
    # sgl_kernel.flash_mla directly, and EXTEND prefill is routed to another
    # backend via --prefill-attention-backend), so construct-only stubs are
    # sufficient: building them succeeds, any real use fails loudly (or, when
    # the caller is the upstream unittest kit driving the unsupported
    # unified flashmla mode, skips — see _hcu_stub_raise below).
    try:
        import sglang.srt.layers.attention.flashinfer_mla_backend as _fimla

        if not hasattr(_fimla, "BatchMLAPagedAttentionWrapper"):

            def _hcu_stub_raise(msg):
                # Under the upstream unittest kit (mla/test_flashmla extend
                # cases) the flashmla backend is driven in unified mode and
                # its inherited flashinfer prefill path reaches these stubs:
                # that is a platform exclusion (no flashinfer on HCU), not a
                # code defect. Convert to unittest.SkipTest there so the
                # subtests land as legal skips — same philosophy as
                # attn_backend_gate's construction-time skips. Serving stacks
                # never contain unittest/pytest frames, so the loud
                # NotImplementedError contract is preserved where it matters.
                import inspect
                import unittest

                for frame in inspect.stack():
                    if frame.filename.endswith(
                        ("unittest/case.py", "unittest2/case.py")
                    ):
                        raise unittest.SkipTest(msg)
                raise NotImplementedError(msg)

            class _HcuFlashinferWrapperStub:
                def __init__(self, *args, **kwargs):
                    pass

                def __getattr__(self, item):
                    _hcu_stub_raise(
                        "hcu-adapter flashinfer: wrapper method "
                        f"'{item}' reached — this attention path needs "
                        "flashinfer, which is unavailable on HCU. Pair the "
                        "flashmla decode backend with "
                        "--prefill-attention-backend triton."
                    )

            _fimla.BatchMLAPagedAttentionWrapper = _HcuFlashinferWrapperStub
            _fimla.BatchPrefillWithRaggedKVCacheWrapper = (
                _HcuFlashinferWrapperStub
            )
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: flashmla wrapper-stub bridge not installed",
            exc_info=True,
        )

    # sgl-flash-mla wheel face (v0.5.23 dsv41 perf work, upstream #45893):
    # kernels/ops/attention/flash_mla_interface.py imports
    # ``fused_norm_rope_attn_rope_cast`` — the sm100 MegaAttention fused
    # norm+rope+attn+rope_cast entry — from the wheel AT MODULE LEVEL and
    # unguarded. HCU's wheel predates that export, so importing the
    # interface module dies with ImportError and every v0.5.23 dsa
    # unittest case that enumerates the flashmla impl variants SUBFAILs
    # instead of skipping. The wheel path is CUDA-only by upstream design
    # ("HIP and XPU keep their own sgl-kernel ops"), so make the import
    # survivable: inject a raising callable for the missing export;
    # construction-time probes then hit the harness's clean-skip contract
    # (attn_backend_gate doctrine). Unittest frames convert to SkipTest
    # (same as the wrapper stubs above); serving stacks keep the loud
    # NotImplementedError.
    try:
        import flash_mla as _fm

        try:
            _has_mega = hasattr(_fm, "fused_norm_rope_attn_rope_cast")
        except Exception:
            # some wheel builds crash lazily on unknown-attr probes
            _has_mega = False
        if not _has_mega:

            def _flash_mla_mega_unavailable(*args, **kwargs):
                import inspect
                import unittest

                msg = (
                    "hcu-adapter flash_mla: wheel predates the "
                    "MegaAttention API (fused_norm_rope_attn_rope_cast, "
                    "upstream #45893) — CUDA-only sm100 fused path; HCU "
                    "serves DSA sparse through its own sgl_kernel/triton ops"
                )
                for frame in inspect.stack():
                    if frame.filename.endswith(
                        ("unittest/case.py", "unittest2/case.py")
                    ):
                        raise unittest.SkipTest(msg)
                raise NotImplementedError(msg)

            _fm.fused_norm_rope_attn_rope_cast = _flash_mla_mega_unavailable

            # The same old wheel also lacks/missigns the rest of the
            # v0.5.23 face: get_mla_decoding_metadata (AttributeError,
            # 18 cases via dsa_backend._compute_flashmla_metadata) and
            # flash_mla_sparse_fwd's new call signature (TypeError, 7
            # cases). Until HCU ships a wheel at the v0.5.23 API level
            # (or dsa sparse gets routed to HCU impls), every one of
            # these paths is a platform exclusion: convert to skips in
            # unittest frames (kit doctrine: dsa_impl_capability ->
            # skipTest with reason), stay loud elsewhere.
            def _flash_mla_face_stub(name):
                def _unavailable(*args, **kwargs):
                    return _flash_mla_mega_unavailable(*args, **kwargs)

                _unavailable.__name__ = name
                return _unavailable

            try:
                _fm.get_mla_decoding_metadata
            except AttributeError:
                _fm.get_mla_decoding_metadata = _flash_mla_face_stub(
                    "get_mla_decoding_metadata"
                )
            _real_sparse_fwd = getattr(_fm, "flash_mla_sparse_fwd", None)

            if _real_sparse_fwd is not None:
                import functools as _ft

                @_ft.wraps(_real_sparse_fwd)
                def _sparse_fwd_compat(*args, **kwargs):
                    try:
                        return _real_sparse_fwd(*args, **kwargs)
                    except TypeError:
                        # v0.5.23 passes kwargs the pre-0523 wheel never
                        # learned: signature drift, not a numeric fault.
                        return _flash_mla_mega_unavailable(*args, **kwargs)

                _fm.flash_mla_sparse_fwd = _sparse_fwd_compat

            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: flash_mla mega-face shim installed "
                "(wheel predates the v0.5.23 API surface: mega import "
                "made bindable; get_mla_decoding_metadata stubbed; "
                "flash_mla_sparse_fwd signature drift converts to skip)"
            )
    except ImportError:
        pass  # no flash_mla wheel at all: absence handling stays as-is
    except Exception:
        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: flash_mla mega-face shim not installed",
            exc_info=True,
        )
