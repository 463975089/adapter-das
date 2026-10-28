"""Compatibility import; implementation lives in engram_gate_triton."""

import sys
from . import engram_gate_triton as _implementation

sys.modules[__name__] = _implementation
