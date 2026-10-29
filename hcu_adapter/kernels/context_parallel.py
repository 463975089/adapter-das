"""Compatibility import; implementation lives in context_parallel_rccl."""

import sys
from . import context_parallel_rccl as _implementation

sys.modules[__name__] = _implementation
