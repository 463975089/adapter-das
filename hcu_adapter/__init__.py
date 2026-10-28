"""HCU compatibility adapter: host dispatch + shared adapters layer.

Package layout (single package; the former framework-neutral
``hcu_adapters`` top-level package was folded in here):

    bootstrap.py               host detection + common->host dispatch
    bridges/                   cross-framework guards + per-host glue
    <域>.py (hcu_env, registry, ...)   framework-neutral mechanism/domain
                                       bridges shared by both hosts
    aiter/ flash_attn/ flashinfer/ fla_hcu/   vendor stubs & vendored kernels

Import-weight contract: importing ``hcu_adapter`` (or
``hcu_adapter.extensions`` for the vLLM plugin seam) must stay light —
no torch / vLLM / sgl_kernel. The vendor-stub registrations that DO pull
torch live in :mod:`hcu_adapter.stubs` (two stub subpackages bind torch
constants at module level) and are triggered exactly once by
``import sgl_kernel`` (see sgl_kernel/__init__.py), preserving the timing:
the meta-path finders install before any sglang module import. ``from <lib>
import <name>`` succeeds at import time (upstream shims bind eagerly), while
*calling* an unimplemented entry raises a loud NotImplementedError naming
the missing kernel; entries with device-agnostic semantics are served by
our triton kernels (currently ``aiter.silu_and_mul``).
"""

from .bootstrap import detect_host

__all__ = ["detect_host"]
