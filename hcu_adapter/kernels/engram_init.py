"""Compatibility import; implementation lives in engram_init_emu."""

import sys
from . import engram_init_emu as _implementation

sys.modules[__name__] = _implementation
