"""Compatibility alias for the common environment registry."""

import sys
from hcu_adapter import hcu_env as _implementation

sys.modules[__name__] = _implementation
