# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first import surface for optional sparse FlashAttention.

Community attention test kits import this module while enumerating backends,
even when the selected Dense/MLA/Mamba case never invokes sparse attention.
Keep collection possible, but fail loudly if the unsupported sparse route is
actually selected instead of pretending that a dense fallback is equivalent.
"""

from __future__ import annotations


def _unsupported(*args, **kwargs):
    raise NotImplementedError(
        "sparse_flash_attn is not available on HCU; select a supported dense "
        "attention backend or install a compatible sparse implementation"
    )


sparse_attn_func = _unsupported
convert_vertical_slash_indexes = _unsupported
convert_vertical_slash_indexes_mergehead = _unsupported

__all__ = [
    "sparse_attn_func",
    "convert_vertical_slash_indexes",
    "convert_vertical_slash_indexes_mergehead",
]
