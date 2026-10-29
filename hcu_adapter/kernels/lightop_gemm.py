"""Compatibility import; implementation lives in gemm_lightop."""

import sys
from . import gemm_lightop as _implementation

sys.modules[__name__] = _implementation
