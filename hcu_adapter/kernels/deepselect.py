"""Compatibility import; implementation lives in deepselect_emu."""

import sys
from . import deepselect_emu as _implementation

sys.modules[__name__] = _implementation
