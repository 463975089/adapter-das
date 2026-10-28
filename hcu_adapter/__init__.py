"""HCU compatibility adapter: host dispatch + shared adapters layer.

Package layout (single package; the former framework-neutral
``hcu_adapter`` top-level package was folded in here):

    bootstrap.py               host detection + common->host dispatch
    bridges/                   cross-framework guards + per-host glue
    <域>.py (hcu_env, registry, ...)   framework-neutral mechanism/domain
                                       bridges shared by both hosts
    aiter/ flash_attn/ flashinfer/ fla_hcu/   vendor stubs & vendored kernels

sglang's HIP code paths import several AMD/NVIDIA-only kernel libraries at
module level. Each stub subpackage here installs a meta-path compatibility
adapter for one of them, so upstream code runs unmodified on HCU:
``from <lib> import <name>`` succeeds at import time, while *calling* an
unimplemented entry raises a loud NotImplementedError naming the missing
kernel. Entries with device-agnostic semantics are served by our triton
kernels (currently ``aiter.silu_and_mul``).

Registration timing is unchanged: ``import sgl_kernel`` triggers
``register_all()`` (see sgl_kernel/__init__.py), so the finders install
before any sglang module import.
"""

from . import flash_attn as _fa
from . import flashinfer as _fi
from . import aiter as _aiter
from . import hadamard as _had

from .bootstrap import detect_host

__all__ = ["detect_host"]


def register_all() -> None:
    """Idempotently install every adapter's meta-path finder."""
    _aiter.register()
    _fi.register()
    _fa.register()
    _had.register()
