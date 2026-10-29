"""Compatibility import; implementation lives in moe_gate_emu."""

import sys
from . import moe_gate_emu as _implementation

sys.modules[__name__] = _implementation
