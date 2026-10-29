"""Compatibility import; implementation lives in grouped_gemm_triton."""

import sys
from . import grouped_gemm_triton as _implementation

sys.modules[__name__] = _implementation
