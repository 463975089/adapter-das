# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Vendored shims for sglang-das `fla/hcu` kernels (branch
main_glm53_flash_feature_cleanup_tx, commit 364fabf).

These kernels are ported verbatim from the sglang-das fork; only their
intra-package imports are rewritten to this adapter package. The three
sglang-internal helpers they pull are shimmed here so the vendor tree
stays importable against official v0.5.21 (whose sglang.srt.utils /
kernels.jit.utils signatures have drifted from the fork):

- ``is_hcu``: the vendored kernels only run on HCU, so the shim is a
  device probe (torch.version.hip) rather than the fork's platform
  registry — same answer on every machine that can execute them.
- ``get_bool_env_var``: plain env lookup (sglang's version logs a
  deprecation warning on v0.5.21 for this name).
- ``is_arch_support_pdl``: probed from sglang.kernels.jit.utils when
  present, else False (PDL-off is the semantics our triton_pdl stripper
  enforces anyway).
- ``torch_release``: version tuple parsed from torch.__version__ (the
  fork's helper is a tuple in sglang.srt.utils.common, which v0.5.21
  does not expose; utils.py compares it with ``>=`` at import time).
"""

from __future__ import annotations

import os

import torch


def is_hcu() -> bool:
    return torch.version.hip is not None


def get_bool_env_var(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in ("true", "1")


def is_arch_support_pdl() -> bool:
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl as _f

        return _f()
    except Exception:
        return False


def _parse_release() -> tuple:
    raw = torch.__version__.split("+")[0]
    parts = []
    for piece in raw.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


torch_release = _parse_release()
