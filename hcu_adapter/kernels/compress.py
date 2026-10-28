"""Compatibility import; implementation lives in compress_emu."""

import sys
from . import compress_emu as _implementation

sys.modules[__name__] = _implementation
