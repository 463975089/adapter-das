"""Compatibility import; implementation lives in rope_triton."""

import sys
from . import rope_triton as _implementation

sys.modules[__name__] = _implementation
