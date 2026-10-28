"""Vendor-stub registration entry — the only torch-importing package front.

Importing this module loads the aiter / flashinfer / flash_attn / hadamard
stub subpackages (two of them bind torch constants at module level, so this
module carries ``import torch`` transitively by design) and registers their
meta-path finders.

Sole caller: ``sgl_kernel/__init__.py``. Keeping these imports OUT of the
package ``__init__`` preserves the plugin-seam contract
``import hcu_adapter.extensions`` stays torch-free (governance §7.2) while
``import sgl_kernel`` still installs every finder before any sglang import.
"""

from . import flash_attn as _fa
from . import flashinfer as _fi
from . import aiter as _aiter
from . import hadamard as _had

__all__ = ["register_all"]


def register_all() -> None:
    """Idempotently install every adapter's meta-path finder."""
    _aiter.register()
    _fi.register()
    _fa.register()
    _had.register()
