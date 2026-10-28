"""Compatibility import; implementation lives in engram_runtime_emu."""

import sys
from . import engram_runtime_emu as _implementation

sys.modules[__name__] = _implementation
