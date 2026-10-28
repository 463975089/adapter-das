"""Compatibility import; implementation lives in weight_scale_emu."""

import sys
from . import weight_scale_emu as _implementation

sys.modules[__name__] = _implementation
