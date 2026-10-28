"""Compatibility import; implementation lives in mla_layout_emu."""

import sys
from . import mla_layout_emu as _implementation

sys.modules[__name__] = _implementation
