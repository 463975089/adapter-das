"""`flash_attn` / FA3 (flash-attention package) adapter for HCU.

sglang's FA3 loading path (`sglang.kernels.ops.attention.flash_attention_v3`)
falls back to `sgl_kernel.flash_attn` / the `flash_attn` package when its JIT
kernel repo is unavailable; dedicated backends import `flash_attn` /
`flash_attn_interface` eagerly.

Vendor-first (§0.5 tier-2): DTK ships a gfx93x `flash_attn` wheel on the
bw/bx NIC images — when it is importable through the plain PathFinder (i.e.
in sys.path), this adapter does NOT intercept and the real wheel serves. Only
when the wheel is absent do lazy stubs cover the imports (`from flash_attn
import x` binds; calling x raises a clear error). The sgl_kernel-level
routing (fa3 backend -> vendor vs in-tree emulation) lives in
`hcu_adapter/fa_vendor.py`, which is what sglang actually calls through.

Covered prefixes: flash_attn, flash_attn_interface.
"""

import importlib.machinery
import sys

from hcu_adapter._base import LibStubFinder, is_registered

_PREFIXES = ("flash_attn", "flash_attn_interface")


def _real_spec(fullname, path=None):
    try:
        return importlib.machinery.PathFinder.find_spec(fullname, path)
    except Exception:
        return None


class _FlashAttnFinder(LibStubFinder):
    def __init__(self):
        super().__init__("flash_attn")

    def find_spec(self, fullname, path=None, target=None):
        for prefix in _PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                real = _real_spec(fullname, path)
                if real is not None:
                    return real  # vendor wheel wins (tier-2 routing)
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None


def register() -> None:
    if not is_registered(_FlashAttnFinder, "flash_attn"):
        sys.meta_path.insert(0, _FlashAttnFinder())
