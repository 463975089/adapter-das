"""Compatibility import; implementation lives in sampling_emu."""

import sys
from . import sampling_emu as _implementation

sys.modules[__name__] = _implementation
