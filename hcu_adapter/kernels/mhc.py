"""Compatibility import; implementation lives in mhc_emu."""

import sys
from . import mhc_emu as _implementation

sys.modules[__name__] = _implementation
