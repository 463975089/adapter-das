"""Compatibility import; implementation lives in quant_emu."""

import sys
from . import quant_emu as _implementation

sys.modules[__name__] = _implementation
