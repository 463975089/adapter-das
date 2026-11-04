# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Attention-backend availability gates at the registry seam.

Two v0.5.21 backends construct fine but die on first use on HCU, and the
upstream attention-kit harness only degrades gracefully for constructor
time failures (attention_methods/dense_attention.py::

    except (AssertionError, ImportError, ModuleNotFoundError):
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

— the same clean skip trtllm_mha earns on non-Blackwell). Gate both at the
official ``ATTENTION_BACKENDS`` registry so the failure mode is a skip, not
a mid-test crash:

- **fa4** — routes through sglang/kernels/ops/attention/flash_attn/cute/
  flash_fwd_sm90.py, whose CUTLASS cute-DSL kernel asserts
  ``is_family_of(Arch.sm_90a)`` at *first forward*; the cute DSL has no
  DTK/HIP build (upstream evidence: dense/test_fa4.py, 36 subcase
  failures, all ``AssertionError: Only SM 9.x is supported``).

- **flashinfer** — upstream's own ``is_flashinfer_available()`` answers
  False on HCU, so its ``from flashinfer import
  BatchPrefillWithRaggedKVCacheWrapper`` block never binds, but the
  registry factory still constructs the backend, which then NameErrors
  on the unbound wrapper name (flashinfer_backend.py:496; upstream
  contract tests failed). flashinfer attention kernels are CUDA-only
  regardless; the flashinfer *sampling* stub is a separate seam and is
  unaffected.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

_GATED = ("fa4", "flashinfer")


def install():
    import torch

    if not torch.version.hip:
        return

    from sglang.srt.layers.attention import attention_registry as areg

    reasons = {
        "fa4": (
            "fa4 requires the CUTLASS cute DSL (SM90a, CUDA-only); "
            "unavailable on HIP/DTK — use the fa3/triton/dsa backends"
        ),
        "flashinfer": (
            "flashinfer attention kernels are CUDA-only and "
            "is_flashinfer_available() is False on HCU; use the "
            "fa3/triton/dsa backends"
        ),
    }

    installed = []
    for name in _GATED:
        orig = areg.ATTENTION_BACKENDS.get(name)
        if orig is None or getattr(orig, "_hcu_backend_gate", False):
            continue

        def _unavailable(runner, _why=reasons[name]):  # noqa: ANN001
            raise ImportError(_why)

        _unavailable._hcu_backend_gate = True
        areg.ATTENTION_BACKENDS[name] = _unavailable
        installed.append(name)

    if not installed:
        return

    import hcu_adapter.registry as _reg

    _reg.register(
        "flash_attention",
        "gfx93x-cuda-only-backends",
        target="seam:sglang...attention_registry.ATTENTION_BACKENDS"
        f"[{'+'.join(installed)}]",
        mechanism=_reg.MECH_SEAM,
        detail="constructor-time ImportError; kit harness skips cleanly",
    )
    LOG.warning(
        "HCU adapter: CUDA-only attention backends gated off at the "
        "registry seam: %s",
        ", ".join(installed),
    )
