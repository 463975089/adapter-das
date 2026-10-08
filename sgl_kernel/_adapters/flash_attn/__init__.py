"""`flash_attn` / FA3 (flash-attention package) adapter for HCU.

sglang's FA3 loading path (`sglang.kernels.ops.attention.flash_attention_v3`)
falls back to `sgl_kernel.flash_attn` / the `flash_attn` package when its JIT
kernel repo is unavailable; dedicated backends import `flash_attn` /
`flash_attn_interface` eagerly. On HCU none are installable (CUDA-only), and
attention runs on the in-tree triton backend instead, so this adapter provides
loud lazy stubs only: imports bind, calls raise a clear error.

Covered prefixes: flash_attn, flash_attn_interface.
"""

import importlib.machinery
import sys

from sgl_kernel._adapters._base import LibStubFinder, is_registered

_PREFIXES = ("flash_attn", "flash_attn_interface")


class _FlashAttnFinder(LibStubFinder):
    def __init__(self):
        super().__init__("flash_attn")

    def find_spec(self, fullname, path=None, target=None):
        for prefix in _PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None


def register() -> None:
    if not is_registered(_FlashAttnFinder, "flash_attn"):
        sys.meta_path.insert(0, _FlashAttnFinder())
