"""Compatibility import; implementation lives in draft_router_triton."""

import sys
from . import draft_router_triton as _implementation

sys.modules[__name__] = _implementation
