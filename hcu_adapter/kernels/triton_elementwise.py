"""Compatibility import; implementation lives in elementwise_triton."""

import sys
from . import elementwise_triton as _implementation

sys.modules[__name__] = _implementation
